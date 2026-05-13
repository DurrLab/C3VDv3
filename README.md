# Colonoscopy 3D deformable video dataset with paired depth from 2D-3D registration and sim2real

<table>
    <tr>
        <td><img src="assets/c2_sigmoid_p2_v1_16.gif" alt="gif 1" width="100%" /></td>
        <td><img src="assets/c0_cecum_t4_v2_16.gif" alt="gif 2" width="100%" /></td>
        <td><img src="assets/c2_trans2_p3_v2_16.gif" alt="gif 3" width="100%" /></td>
        <td><img src="assets/c2_trans1_p1_v1_16.gif" alt="gif 4" width="100%" /></td>
    </tr>
</table>

This repository contains the rendering code used in *Colonoscopy 3D deformable video dataset with paired depth from 2D-3D registration and sim2real*. Visit the [project webpage](https://durrlab.github.io/C3VD/) to learn more about this work.

<video autoplay loop muted playsinline width="100%">
  <source src="assets/aligned_prev_final.mp4" type="video/mp4">
</video>


## Prerequisites
### Software
* Ubuntu 20.04
* CMake>=3.5
* Nvidia Device Drivers>=450
* Nvidia CUDA>=11.1
### Hardware
* NVIDIA GPU of Compute Capability 5.0 (Maxwell) or higher.

## Build Instructions
1. Install required third-party libraries:
```sudo apt install -y freeglut3-dev libglew-dev```

2. Pull the repository and associated submodules:
```
git clone https://github.com/DurrLab/C3VD.git
cd C3VD
git submodule init
git submodule update
```
3. Build the submodules and main executables:
```
mkdir build
cd build
cmake ..
make -j8
```

## Usage
Before running any of the programs, create a new working directory for each video sequence, organized as follows:
```bash
.
└── SAMPLE_DIR/       # working directory for the given video sequences
    ├── config.ini    # parameter file
    ├── model.obj     # ground truth 3D model
    ├── model.mtl     # ground truth 3D model material
    ├── pose.txt      # robot pose log; one pose per line, formatted <time in seconds> <homogenous pose in column-major form>
    ├── mask.png      # binary corner mask for Olympus endoscopes
    ├── rgb/          # rgb image folder
    │   ├── 0000.png         
    │   ├── 0001.png
    │   │   ...
    │   └── N-1.png
    ├── edges/        # GAN-predicted edge image folder
    │   ├── 0000.png         
    │   ├── 0001.png
    │   │   ...
    │   └── N-1.png
    ├── results/      # registration results folder
    └── render/       # gt rendering output folder
```

### Ground Truth Rendering
Update the modelTransform parameter in the configuration file to the result from the registration program. Then, run the ground truth rendering program:
```
./c3vd rendergt <SAMPLE_DIR>
```
Rendered ground truth files are saved in the *render* folder.

### 3D Colon Deformation

This repository includes a deformation generation workflow for simulating colon deformations and rendering them with the C3VD framework. Note: An existing C3VD rendered output is recommended before running the deformation pipeline and should have the minimal following structure:

```bash
.
└── reference_dir/           # reference directory for the given video sequences
    ├── rgb/                 # rgb image folder
    │   ├── 0000.png         
    │   ├── 0001.png
    │   │   ...
    │   └── N-1.png
    ├── coverage_mesh.obj    # Undeformed Reference Mesh
    └── pose.txt             # 4×4 homogeneous transformation matrix in row-major format
```

#### Setup

Install the Python dependencies for the deformation scripts with:
```
python -m pip install -r deformation/requirements.txt
```

#### Configuration

Create a YAML config file for your deformation. Two template configs are provided in [deformation/config/templates/](deformation/config/templates/):

**Gaussian Deformation (Waves)**

Use this for simulating peristaltic waves or other wave-based deformations:

```yaml
geometry: c3vd_or_c3vdv2_geometry
reference_dir: /path/to/reference/contents
centerline_path: /path/to/extracted/centerline.npy
output_root: /path/to/output_root

enable_gaussian: true
enable_centerline_warp: false

waves:
  - A: 0.6                    # amplitude (mm)
    sigma_frac: 0.10          # gaussian width as fraction of centerline length
    velocity_cm_s: 2.0        # wave propagation speed (cm/s)
    start_delay_s: 0.0        # delay before the wave starts (seconds)

fps: 29.97
taubin_iterations: 12         # mesh smoothing iterations
subdivision_iterations: 0     # mesh subdivision
```

**Centerline-based Warp**

```yaml
geometry: c3vd_or_c3vdv2_geometry
reference_dir: /path/to/reference/contents
centerline_path: /path/to/extracted/centerline.npy
output_root: /path/to/output_root

enable_gaussian: false
enable_centerline_warp: true

transform_mode: warp             # or: shift, linear_shift, exp_tail_warp
transform_params:
  amplitude: 8.0                 # peak displacement (mm)
  axis: auto                     # or [x, y, z] direction vector
  phase: 0.0                     # initial phase offset

fps: 29.97
save_new_centerline: true
taubin_iterations: 12
```

**Required Fields:**
- `geometry`: A unique identifier for this deformation (used in output naming)
- `reference_dir`: Path to the original mesh directory (should contain `rgb/`, `coverage_mesh.obj`, `pose.txt`)
- `centerline_path`: Path to the centerline NumPy array (`.npy` format, shape [N, 3])
- `output_root`: Root output directory where results are written

#### Running the Generator

Run the deformation pipeline with a single config:
```bash
cd /path/to/C3VD_deformation
./deformation/run_deformation_generator.sh --config deformation/config/my_config.yaml
```

Or process all configs in a directory:
```bash
./deformation/run_deformation_generator.sh --config-dir deformation/config/
```

The script will:
1. Generate the deformed geometry mesh
2. Copy required input files (RGB, edges, mask, pose log) from the reference
3. Run the C3VD renderer on the deformed sequence
4. Compute render error metrics

#### Output Structure

Results are organized as:
```
<c3vd_input_path>/<geometry>/
├── render/
│   ├── depth/
│   ├── diffuse/  
│   ├── normals/
│   ├── occlusion/
│   ├── optical_flow/ 
│   ├── coverage_mesh.obj                   
│   ├── pose.txt                            
│   ├── world_vertex_positions.bin          # per-frame vertex positions in world coordinates (binary)
│   ├── world_vertex_normals.bin            # per-frame vertex normals inworld coordinates (binary)
├── render_comparison/            # error metrics vs. reference
└── config.yaml                   # copy of the config used
```

#### Preview (Optional)

Before running the full pipeline, preview the deformation with:
```bash
python deformation/scripts/preview_gaussian.py --config deformation/config/my_config.yaml
python deformation/scripts/preview_centerline_warp.py --config deformation/config/my_config.yaml
```

These scripts visualize the mesh deformation in real-time using Open3D.

## Sample Video Sequence
A sample raw video sequence from the dataset is available for download [HERE](https://drive.google.com/file/d/1Ddeq5Dm4tx7cMRTZBu3CN3otsGu2_kY1/view?usp=sharing). Once uncompressed, the folder is ready to be run by the programs.

<!-- 
## Reference
If you find our work useful in your research, please consider citing our paper:
```
placeholder
```
-->
## License
This work is licensed under CC BY 4.0
