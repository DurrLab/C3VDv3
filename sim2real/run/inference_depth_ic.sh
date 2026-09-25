#!/usr/bin/env bash
# Depth IC-LoRA inference launcher.
#
# Runs the depth-conditioned image-to-video pipeline end to end:
#
#   [0/3] Filter the input clips JSON to those that have a paired first-frame
#         reference image in FIRST_FRAME_MAPPING.
#   [1/3] Stage inference inputs (per-clip depth PNG sequences with optional
#         spatial filter + inverse-depth percentile stretch + gamma) via
#         scripts/prepare_validation_inputs_depth.py.
#   [2/3] Cache Gemma + connector text embeddings on GPU 0 via
#         scripts/precompute_validation_prompts.py (or reuse an existing cache).
#   [3/3] Launch packages/ltx-trainer/scripts/inference_edge_ic.py for every
#         (clip x reference image x seed) tuple, sharded across visible GPUs.
#
# inference_edge_ic.py reads the depth PNGs directly via
# --reference-frames-dir; no MP4 roundtrip is performed for the depth signal.
#
# Required env vars:
#   REPO_ROOT              Path to the upstream LTX-2 working tree
#                          (defaults to ../LTX-2 relative to this file).
#   BASE_MODEL_PATH        Path to the base LTX-2 colonoscopy checkpoint
#                          (.safetensors).
#   TEXT_ENCODER_PATH      Path to the Gemma encoder directory.
#   DEPTH_LORA_DIR         Directory holding lora_weights_step_*.safetensors.
#   CLIPS_JSON             Input clips JSON (frame-list metadata).
#   FIRST_FRAME_MAPPING    JSON mapping clip names -> reference image
#                          filename(s) used as the first-frame appearance anchor.
#   REFERENCE_IMAGES_DIR   Directory holding the reference appearance images.
#   OUTPUT_ROOT            Where to write inputs/, videos/, generated_frames/.
#
# Optional:
#   NUM_CLIPS=9 NUM_FRAMES=89 FPS=16 WIDTH=512 HEIGHT=512
#   CLIP_SEED=42 SEEDS=42 DOMAIN=synthetic DEPTH_SUFFIX=_depth
#   NUM_INFERENCE_STEPS=30 GUIDANCE_SCALE=4.0 LORA_SCALE=1.0
#   STG_SCALE=1.0 STG_MODE=stg_v STG_BLOCKS=29
#   REFERENCE_RESIZE_MODE=stretch CONDITION_IMAGE_RESIZE_MODE=stretch
#   REFERENCE_STRENGTH=1.0 CONDITION_IMAGE_STRENGTH=1.0
#   DEPTH_PREPROCESS_MODE=c3vd_inverse_depth_keep_orient
#   DEPTH_FILTER_TYPE=gaussian
#   DEPTH_GAUSSIAN_SIGMA_256=0.5 DEPTH_MEDIAN_KERNEL_SIZE=3
#   DEPTH_BILATERAL_DIAMETER=9 DEPTH_BILATERAL_SIGMA_COLOR=0.1
#   DEPTH_BILATERAL_SIGMA_SPACE_256=3.0 DEPTH_GAMMA=0.7
#   INVERSE_DEPTH_PCT_LOW=2 INVERSE_DEPTH_PCT_HIGH=98 INVERSE_DEPTH_EPS=1
#   USE_FIRST_FRAME=1 INCLUDE_REFERENCE_IN_OUTPUT=0
#   INCLUDE_SOURCE_IN_OUTPUT=0 SAVE_GENERATED_FRAMES=1
#   REUSE_PROMPT_CACHE=1 LOAD_TEXT_ENCODER_IN_8BIT=1
#   MAX_REFS_PER_CLIP=0 (0 = use all references per clip)
#   STATIC_PROMPT_CACHE_DIR  Path to shared prompt cache; if set, the script
#                            reuses it across runs.
#   MAX_PARALLEL  Cap on concurrent per-GPU inference jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${RELEASE_ROOT}/LTX-2}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: LTX-2 working tree not found at ${REPO_ROOT}." >&2
  echo "       Run 'bash setup_overlay.sh' first, or set REPO_ROOT explicitly." >&2
  exit 1
fi
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:?Set TEXT_ENCODER_PATH}"
DEPTH_LORA_DIR="${DEPTH_LORA_DIR:?Set DEPTH_LORA_DIR}"
DEPTH_LORA_CHECKPOINT="${DEPTH_LORA_CHECKPOINT:-latest}"
CLIPS_JSON="${CLIPS_JSON:?Set CLIPS_JSON}"
FIRST_FRAME_MAPPING="${FIRST_FRAME_MAPPING:?Set FIRST_FRAME_MAPPING}"
REFERENCE_IMAGES_DIR="${REFERENCE_IMAGES_DIR:?Set REFERENCE_IMAGES_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

if [[ "${DEPTH_LORA_CHECKPOINT}" == "latest" ]]; then
  if [[ ! -d "${DEPTH_LORA_DIR}" ]]; then
    echo "ERROR: DEPTH_LORA_DIR does not exist: ${DEPTH_LORA_DIR}" >&2
    exit 1
  fi
  DEPTH_LORA_CHECKPOINT="$(${PYTHON_BIN} -c "
import re, sys
from pathlib import Path
d = Path('${DEPTH_LORA_DIR}')
cands = []
for p in d.glob('lora_weights_step_*.safetensors'):
    m = re.search(r'step_(\d+)', p.name)
    if m:
        cands.append((int(m.group(1)), p))
if not cands:
    sys.exit('no lora_weights_step_*.safetensors under ' + str(d))
cands.sort()
print(cands[-1][1])
")"
  echo "Resolved DEPTH_LORA_CHECKPOINT=latest -> ${DEPTH_LORA_CHECKPOINT}"
fi

NUM_CLIPS="${NUM_CLIPS:-9}"
NUM_FRAMES="${NUM_FRAMES:-89}"
FPS="${FPS:-16}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
CLIP_SEED="${CLIP_SEED:-42}"
SEEDS="${SEEDS:-42}"
DOMAIN="${DOMAIN:-synthetic}"
DEPTH_SUFFIX="${DEPTH_SUFFIX:-_depth}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
LORA_SCALE="${LORA_SCALE:-1.0}"
STG_SCALE="${STG_SCALE:-1.0}"
STG_MODE="${STG_MODE:-stg_v}"
STG_BLOCKS="${STG_BLOCKS:-29}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-worst quality, inconsistent motion, blurry, jittery, distorted}"
INCLUDE_REFERENCE_IN_OUTPUT="${INCLUDE_REFERENCE_IN_OUTPUT:-0}"
INCLUDE_SOURCE_IN_OUTPUT="${INCLUDE_SOURCE_IN_OUTPUT:-0}"
SAVE_GENERATED_FRAMES="${SAVE_GENERATED_FRAMES:-1}"
REUSE_PROMPT_CACHE="${REUSE_PROMPT_CACHE:-1}"
MAX_REFS_PER_CLIP="${MAX_REFS_PER_CLIP:-0}"
STATIC_PROMPT_CACHE_DIR="${STATIC_PROMPT_CACHE_DIR:-}"
LOAD_TEXT_ENCODER_IN_8BIT="${LOAD_TEXT_ENCODER_IN_8BIT:-1}"
USE_FIRST_FRAME="${USE_FIRST_FRAME:-1}"
REFERENCE_RESIZE_MODE="${REFERENCE_RESIZE_MODE:-stretch}"
CONDITION_IMAGE_RESIZE_MODE="${CONDITION_IMAGE_RESIZE_MODE:-stretch}"
REFERENCE_STRENGTH="${REFERENCE_STRENGTH:-1.0}"
CONDITION_IMAGE_STRENGTH="${CONDITION_IMAGE_STRENGTH:-1.0}"
DEPTH_FILTER_TYPE="${DEPTH_FILTER_TYPE:-gaussian}"
DEPTH_GAUSSIAN_SIGMA_256="${DEPTH_GAUSSIAN_SIGMA_256:-0.5}"
DEPTH_MEDIAN_KERNEL_SIZE="${DEPTH_MEDIAN_KERNEL_SIZE:-3}"
DEPTH_BILATERAL_DIAMETER="${DEPTH_BILATERAL_DIAMETER:-9}"
DEPTH_BILATERAL_SIGMA_COLOR="${DEPTH_BILATERAL_SIGMA_COLOR:-0.1}"
DEPTH_BILATERAL_SIGMA_SPACE_256="${DEPTH_BILATERAL_SIGMA_SPACE_256:-3.0}"
DEPTH_GAMMA="${DEPTH_GAMMA:-0.7}"
DEPTH_PREPROCESS_MODE="${DEPTH_PREPROCESS_MODE:-c3vd_inverse_depth_keep_orient}"
INVERSE_DEPTH_PCT_LOW="${INVERSE_DEPTH_PCT_LOW:-2}"
INVERSE_DEPTH_PCT_HIGH="${INVERSE_DEPTH_PCT_HIGH:-98}"
INVERSE_DEPTH_EPS="${INVERSE_DEPTH_EPS:-1}"

if [[ -z "${CAPTION_POOL+x}" ]]; then
  CAPTION_POOL=(
    "Real Colonoscopy Image, White Light Imaging"
  )
fi

CKPT_STEM="$(basename "${DEPTH_LORA_CHECKPOINT}" .safetensors)"
MODE_TAG="i2v"
if [[ "${USE_FIRST_FRAME}" != "1" ]]; then MODE_TAG="t2v"; fi

OUTPUT_ROOT="${OUTPUT_ROOT}/${CKPT_STEM}_${MODE_TAG}"
INPUTS_DIR="${OUTPUT_ROOT}/inputs"
VIDEOS_DIR="${OUTPUT_ROOT}/videos"
GENERATED_FRAMES_DIR="${OUTPUT_ROOT}/generated_frames"
PROMPT_CACHE_DIR="${INPUTS_DIR}/prompt_cache"
if [[ -n "${STATIC_PROMPT_CACHE_DIR}" ]]; then
  PROMPT_CACHE_DIR="${STATIC_PROMPT_CACHE_DIR}"
fi
MANIFEST="${INPUTS_DIR}/manifest.json"

for path in "${BASE_MODEL_PATH}" "${DEPTH_LORA_CHECKPOINT}" "${CLIPS_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${TEXT_ENCODER_PATH}" ]]; then
  echo "ERROR: text encoder dir not found: ${TEXT_ENCODER_PATH}" >&2
  exit 1
fi
if [[ "${USE_FIRST_FRAME}" == "1" ]]; then
  if [[ ! -e "${FIRST_FRAME_MAPPING}" ]]; then
    echo "ERROR: first-frame mapping not found: ${FIRST_FRAME_MAPPING}" >&2
    exit 1
  fi
  if [[ ! -d "${REFERENCE_IMAGES_DIR}" ]]; then
    echo "ERROR: reference images dir not found: ${REFERENCE_IMAGES_DIR}" >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_ROOT}" "${VIDEOS_DIR}" "${PROMPT_CACHE_DIR}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${SLURM_GPUS_ON_NODE}"
else
  NUM_GPUS="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: No visible GPUs." >&2
  exit 1
fi
MAX_PARALLEL="${MAX_PARALLEL:-${NUM_GPUS}}"
if [[ "${MAX_PARALLEL}" -gt "${NUM_GPUS}" ]]; then MAX_PARALLEL="${NUM_GPUS}"; fi

echo "Depth IC inference (mode=${MODE_TAG})"
echo "  REPO_ROOT                       = ${REPO_ROOT}"
echo "  BASE_MODEL_PATH               = ${BASE_MODEL_PATH}"
echo "  DEPTH_LORA_CHECKPOINT                = ${DEPTH_LORA_CHECKPOINT}"
echo "  TEXT_ENCODER_PATH               = ${TEXT_ENCODER_PATH}"
echo "  CLIPS_JSON                      = ${CLIPS_JSON}"
echo "  FIRST_FRAME_MAPPING             = ${FIRST_FRAME_MAPPING}"
echo "  REFERENCE_IMAGES_DIR           = ${REFERENCE_IMAGES_DIR}"
echo "  OUTPUT_ROOT                     = ${OUTPUT_ROOT}"
echo "  PROMPT_CACHE_DIR                = ${PROMPT_CACHE_DIR}"
echo "  REUSE_PROMPT_CACHE              = ${REUSE_PROMPT_CACHE}"
echo "  NUM_CLIPS/FRAMES                = ${NUM_CLIPS} clips, ${NUM_FRAMES} frames @ ${FPS} fps"
echo "  WIDTH x HEIGHT                  = ${WIDTH} x ${HEIGHT}"
echo "  DEPTH_PREPROCESS_MODE           = ${DEPTH_PREPROCESS_MODE}"
echo "  DEPTH_FILTER_TYPE               = ${DEPTH_FILTER_TYPE}"
echo "  DEPTH_GAMMA                     = ${DEPTH_GAMMA}"
echo "  INVERSE_DEPTH_PCT (low/high)    = ${INVERSE_DEPTH_PCT_LOW}/${INVERSE_DEPTH_PCT_HIGH}"
echo "  REFERENCE_RESIZE_MODE           = ${REFERENCE_RESIZE_MODE}"
echo "  REFERENCE_STRENGTH              = ${REFERENCE_STRENGTH}"
echo "  CONDITION_IMAGE_STRENGTH        = ${CONDITION_IMAGE_STRENGTH}"
echo "  NUM_GPUS                        = ${NUM_GPUS} (MAX_PARALLEL=${MAX_PARALLEL})"

PREP_CLIPS_JSON="${CLIPS_JSON}"
if [[ "${USE_FIRST_FRAME}" == "1" ]]; then
  MAPPED_CLIPS_JSON="${INPUTS_DIR}/clips_mapped_for_validation.json"
  mkdir -p "${INPUTS_DIR}"
  echo
  echo "[0/3] Filtering validation clips to mapped first-frame entries..."
  "${PYTHON_BIN}" - "${CLIPS_JSON}" "${FIRST_FRAME_MAPPING}" "${MAPPED_CLIPS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

clips_path = Path(sys.argv[1])
mapping_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

with open(clips_path) as f:
    clips = json.load(f)
with open(mapping_path) as f:
    mapping = json.load(f)

mapped_names = {
    m["video_name"]
    for m in mapping
    if m.get("video_name") and m.get("reference_image")
}
filtered = [clip for clip in clips if clip.get("name") in mapped_names]

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"  total clips in original JSON = {len(clips)}")
print(f"  mapped video names           = {len(mapped_names)}")
print(f"  clips retained for prep      = {len(filtered)}")
print(f"  filtered clips JSON          = {output_path}")

if not filtered:
    sys.exit("ERROR: no clips from CLIPS_JSON matched FIRST_FRAME_MAPPING video_name entries")
PY
  PREP_CLIPS_JSON="${MAPPED_CLIPS_JSON}"
fi

echo
echo "[1/3] Preparing validation inputs..."
"${PYTHON_BIN}" "${RELEASE_ROOT}/scripts/prepare_validation_inputs_depth.py" \
  --clips-json "${PREP_CLIPS_JSON}" \
  --output-dir "${INPUTS_DIR}" \
  --num-clips "${NUM_CLIPS}" \
  --num-frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --domain "${DOMAIN}" \
  --seed "${CLIP_SEED}" \
  --depth-suffix "${DEPTH_SUFFIX}" \
  --depth-preprocess-mode "${DEPTH_PREPROCESS_MODE}" \
  --depth-filter-type "${DEPTH_FILTER_TYPE}" \
  --inverse-depth-percentile-low "${INVERSE_DEPTH_PCT_LOW}" \
  --inverse-depth-percentile-high "${INVERSE_DEPTH_PCT_HIGH}" \
  --inverse-depth-eps "${INVERSE_DEPTH_EPS}" \
  --gaussian-sigma-256 "${DEPTH_GAUSSIAN_SIGMA_256}" \
  --median-kernel-size "${DEPTH_MEDIAN_KERNEL_SIZE}" \
  --bilateral-diameter "${DEPTH_BILATERAL_DIAMETER}" \
  --bilateral-sigma-color "${DEPTH_BILATERAL_SIGMA_COLOR}" \
  --bilateral-sigma-space-256 "${DEPTH_BILATERAL_SIGMA_SPACE_256}" \
  --depth-gamma "${DEPTH_GAMMA}" \
  --prompt-cache-dir "${PROMPT_CACHE_DIR}" \
  --caption-override-pool "${CAPTION_POOL[@]}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not written: ${MANIFEST}" >&2
  exit 1
fi

echo
mapfile -t EXPECTED_PROMPT_CACHE_FILES < <(
  "${PYTHON_BIN}" - "${PROMPT_CACHE_DIR}" "${CAPTION_POOL[@]}" <<'PY'
import re
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
for prompt in dict.fromkeys(sys.argv[2:]):
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    print(cache_dir / f"{slug}.pt")
PY
)

PROMPT_CACHE_COMPLETE=1
for cache_file in "${EXPECTED_PROMPT_CACHE_FILES[@]}"; do
  if [[ ! -f "${cache_file}" ]]; then
    PROMPT_CACHE_COMPLETE=0
    break
  fi
done

TE_8BIT_FLAG=()
if [[ "${LOAD_TEXT_ENCODER_IN_8BIT}" == "1" ]]; then
  TE_8BIT_FLAG+=(--load-text-encoder-in-8bit)
fi

if [[ "${REUSE_PROMPT_CACHE}" == "1" && "${PROMPT_CACHE_COMPLETE}" == "1" ]]; then
  echo "[2/3] Reusing prompt embeddings cache; skipping Gemma loading."
  for cache_file in "${EXPECTED_PROMPT_CACHE_FILES[@]}"; do
    echo "  found cache = $(basename "${cache_file}")"
  done
else
  echo "[2/3] Precomputing prompt embeddings on GPU 0..."
  CUDA_VISIBLE_DEVICES="0" \
  "${PYTHON_BIN}" "${RELEASE_ROOT}/scripts/precompute_validation_prompts.py" \
    --checkpoint "${BASE_MODEL_PATH}" \
    --text-encoder-path "${TEXT_ENCODER_PATH}" \
    --output-dir "${PROMPT_CACHE_DIR}" \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --prompts "${CAPTION_POOL[@]}" \
    "${TE_8BIT_FLAG[@]}"
fi

echo
echo "[3/3] Launching inference..."

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"

# Expand manifest into one row per (clip x reference image) pair using
# FIRST_FRAME_MAPPING (in T2V mode, emit one row per clip with no first-frame
# condition).
mapfile -t MANIFEST_ROWS < <(
  "${PYTHON_BIN}" -c "
import json, os, sys

manifest = json.load(open('${MANIFEST}'))
use_ff = '${USE_FIRST_FRAME}' == '1'
mapping = json.load(open('${FIRST_FRAME_MAPPING}')) if use_ff else []
ref_by_video = {m['video_name']: m['reference_image'] for m in mapping}
ref_dir = '${REFERENCE_IMAGES_DIR}'
max_refs = int('${MAX_REFS_PER_CLIP}')

for e in manifest:
    name = e['clip_name']
    if use_ff:
        refs = ref_by_video.get(name)
        if not refs:
            sys.stderr.write(f'Skipping unmapped clip: {name}\n')
            continue
        if isinstance(refs, str):
            refs = [refs]
        if max_refs > 0:
            refs = refs[:max_refs]
        for ref in refs:
            ref_path = ref if os.path.isabs(ref) else os.path.join(ref_dir, ref)
            ref_stem = os.path.splitext(os.path.basename(ref))[0]
            print('\t'.join([
                name,
                ref_path,
                e['depth_frames_dir'],
                e['caption'],
                e.get('cached_condition_path', ''),
                e.get('caption_style', ''),
                ref_stem,
                e.get('source_rgb_frames_dir', ''),
            ]))
    else:
        print('\t'.join([
            name,
            '',
            e['depth_frames_dir'],
            e['caption'],
            e.get('cached_condition_path', ''),
            e.get('caption_style', ''),
            '',
            e.get('source_rgb_frames_dir', ''),
        ]))
"
)

if [[ "${#MANIFEST_ROWS[@]}" -eq 0 ]]; then
  echo "ERROR: no manifest rows after mapping filter." >&2
  exit 1
fi

running_jobs=0
job_index=0

REFERENCE_FLAG=()
if [[ "${INCLUDE_REFERENCE_IN_OUTPUT}" == "1" ]]; then
  REFERENCE_FLAG+=(--include-reference-in-output)
fi

launch_job() {
  local clip_name="$1" first_frame="$2" depth_dir="$3" caption="$4" cached_cond="$5" style="$6" ref_stem="$7" source_rgb_dir="$8" seed="$9"
  local gpu_index=$((job_index % MAX_PARALLEL))
  local suffix=""
  if [[ -n "${ref_stem}" ]]; then suffix="__ref${ref_stem}"; fi
  if [[ -n "${style}" ]]; then suffix="${suffix}__${style}"; fi
  local output_path="${VIDEOS_DIR}/${clip_name}${suffix}__${MODE_TAG}__seed${seed}.mp4"
  local frame_output_dir="${GENERATED_FRAMES_DIR}/${clip_name}${suffix}__${MODE_TAG}__seed${seed}"

  echo
  echo "Launching job ${job_index}: clip=${clip_name} ref=${ref_stem:-<none>} seed=${seed} gpu=${gpu_index}"

  if [[ -z "${cached_cond}" || ! -f "${cached_cond}" ]]; then
    echo "ERROR: cached condition file missing: ${cached_cond}" >&2
    exit 1
  fi

  local COND_FLAG=()
  if [[ "${USE_FIRST_FRAME}" == "1" ]]; then
    if [[ ! -f "${first_frame}" ]]; then
      echo "ERROR: reference image missing: ${first_frame}" >&2
      exit 1
    fi
    COND_FLAG+=(--condition-image "${first_frame}")
  fi

  local SOURCE_FLAG=()
  if [[ "${INCLUDE_SOURCE_IN_OUTPUT}" == "1" ]]; then
    if [[ -z "${source_rgb_dir}" || ! -d "${source_rgb_dir}" ]]; then
      echo "ERROR: source RGB frames dir missing: ${source_rgb_dir}" >&2
      exit 1
    fi
    SOURCE_FLAG+=(--source-frames-dir "${source_rgb_dir}")
  fi

  local FRAME_OUTPUT_FLAG=()
  if [[ "${SAVE_GENERATED_FRAMES}" == "1" ]]; then
    FRAME_OUTPUT_FLAG+=(--output-frames-dir "${frame_output_dir}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu_index}" \
  ${PYTHON_BIN} packages/ltx-trainer/scripts/inference_edge_ic.py \
    --checkpoint "${BASE_MODEL_PATH}" \
    --text-encoder-path "${TEXT_ENCODER_PATH}" \
    --lora-path "${DEPTH_LORA_CHECKPOINT}" \
    --lora-scale "${LORA_SCALE}" \
    --prompt "${caption}" \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --cached-condition-path "${cached_cond}" \
    "${COND_FLAG[@]}" \
    --reference-frames-dir "${depth_dir}" \
    --width "${WIDTH}" --height "${HEIGHT}" \
    --num-frames "${NUM_FRAMES}" --frame-rate "${FPS}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --stg-scale "${STG_SCALE}" --stg-mode "${STG_MODE}" --stg-blocks "${STG_BLOCKS}" \
    --reference-strength "${REFERENCE_STRENGTH}" \
    --reference-resize-mode "${REFERENCE_RESIZE_MODE}" \
    --condition-image-resize-mode "${CONDITION_IMAGE_RESIZE_MODE}" \
    --condition-image-strength "${CONDITION_IMAGE_STRENGTH}" \
    --seed "${seed}" \
    --skip-audio \
    "${REFERENCE_FLAG[@]}" \
    "${SOURCE_FLAG[@]}" \
    "${FRAME_OUTPUT_FLAG[@]}" \
    --output "${output_path}" &

  running_jobs=$((running_jobs + 1))
  job_index=$((job_index + 1))

  if [[ "${running_jobs}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
}

for row in "${MANIFEST_ROWS[@]}"; do
  IFS=$'\t' read -r clip_name first_frame depth_dir caption cached_cond style ref_stem source_rgb_dir <<< "${row}"
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed// /}"
    [[ -z "${seed}" ]] && continue
    launch_job "${clip_name}" "${first_frame}" "${depth_dir}" "${caption}" "${cached_cond}" "${style}" "${ref_stem}" "${source_rgb_dir}" "${seed}"
  done
done

wait

echo
echo "Depth IC inference finished (mode=${MODE_TAG}). Outputs under: ${VIDEOS_DIR}"
