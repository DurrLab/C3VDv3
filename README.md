# Colonoscopy 3D video dataset with paired depth from 2D-3D registration

![banner](https://durrlab.github.io/C3VD/assets/img/sample.gif)

This repository contains the registration and rendering code used in *Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration*. Visit the [project webpage](https://durrlab.github.io/C3VD/) to learn more about this work.

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
The build will compile three executable files placed in the *bin* folder:
- *align*: launches a Graphical User Interface (GUI) for manually initializing the 3D model position
- *register*: registers the 3D model to the target depth frames
- *rendergt*: renders and saves ground truth depth, surface normals, optical flow, and occlusion frames for every frame in the video sequence. Additionally, outputs poses for every frame and a coverage map for the entire video sequence. 

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
### Manual Initialization
The *align* program launches a Graphical User Interface (GUI) that allows users to manually perturb the model position to roughly align it with the video sequence. Video frames are overlayed with renderings of the 3D model, and the camera pose is updated as the video is navigated. The following parameters should be defined in the parameter file (config.ini):
- Omnidirectional camera intrinsics: *width*, *height*, *cx*, *cy*, *ao*, *a1*, *a2*, *a3*, *a4*, *c*, *d*, *e*
- *Acal*: Robot pose retained from the handeye calibration (homogenous, column-major) 
- *Bcal*: Camera pose retained from the handeye calibration (homogenous, column-major)
- *X*: Handeye calibration matrix  (homogenous, column-major)
- *modelTransform*: Initial model transform with 6 values: X-Y-Z axis rotation in radians and X-Y-Z translation in millimeters
- *poseStartTime*: Temporal offset (in seconds) to synchronize the pose log with the video sequence. Frame 0 is paired with pose at time *poseStartTime* in the pose log

To run the program:
```
./c3vd align <SAMPLE_DIR>
```
Parameters can be manipulated using inputs on the GUI window or keyboard. Press 'i' to print the keyboard input key to the terminal window.

<p align="center">
  <img src="https://github.com/DurrLab/C3VD/blob/gh-pages/assets/img/alignmentGui.png" alt="ply" width=320/>
</p>

### Registration
After updating the <modelTransform></modelTransform> parameters in the configuration file with the model transform values from the alignment GUI, an optimization can be run to fine-tune the video alignment. In addition to the configuration parameters listed above, the following parameters should be added to the configuration file before running the registration program:
- *deltaR*: +/- parameter space bounds for rotation components of model position (radians)
- *deltaT*: +/- parameter space bounds for translation components of model position (millimeters)
- *popSize*: Population size for CMAES optimization
- *sigma*: Search sigma for CMAES optimization
- *K*: number of target frames to sample from the video sequence for registration

To run the program:
```
./c3vd register <SAMPLE_DIR>
```
Once the registration is complete, the optimized model transform is printed to the terminal window. Initial and final alignment images are saved in the *results* subdirectory.

### Ground Truth Rendering
Update the modelTransform parameter in the configuration file to the result from the registration program. Then, run the ground truth rendering program:
```
./c3vd rendergt <SAMPLE_DIR>
```
Rendered ground truth files are saved in the *render* folder.

### 3D Colon Deformation

This repository includes a deformation generation workflow for simulating colon deformations (e.g., peristalsis, polyp compression) and rendering them with the C3VD registration framework. The pipeline is controlled through YAML configuration files and supports two deformation types: **Gaussian** (wave-based) and **centerline-based warping**.

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
geometry: my_geometry_name
reference_dir: /path/to/reference/mesh/directory
centerline_path: /path/to/geometry.npy
output_root: /path/to/output_root

enable_gaussian: true
enable_centerline_warp: false

waves:
  - A: 0.6                    # amplitude (mm)
    sigma_frac: 0.10          # gaussian width as fraction of centerline length
    velocity_cm_s: 2.0        # wave propagation speed (cm/s)
    start_delay_s: 0.0        # when the wave starts (seconds)

fps: 29.97
taubin_iterations: 12         # mesh smoothing iterations
subdivision_iterations: 0     # mesh subdivision
```

**Centerline-based Warp**

```yaml
geometry: my_geometry_name
reference_dir: /path/to/reference/mesh/directory
centerline_path: /path/to/geometry.npy
output_root: /path/to/output_root

enable_gaussian: false
enable_centerline_warp: true

transform_mode: warp             # or: shift, linear_shift, exp_tail_warp
transform_params:
  amplitude: 8.0                 # peak displacement (mm)
  axis: auto                     # or [x, y, z] direction vector
  phase: 0.0                     # initial phase offset

fps: 30
save_new_centerline: true
taubin_iterations: 12
```

**Required Fields:**
- `geometry`: A unique identifier for this deformation (used in output naming)
- `reference_dir`: Path to the original mesh directory (should contain `model.obj`, `model.mtl`, etc.)
- `centerline_path`: Path to the centerline NumPy file (`.npy` format, shape [N, 3])
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
├── model.obj, model.mtl         # deformed mesh (frame 0)
├── vertex_positions.bin          # per-frame vertex positions (binary)
├── depth/, normals/, etc.        # rendered ground truth
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

## Visualize Coverage Map
To visualize a coverage map similar to Figure 9 in the manuscript, open the coverage_map.obj output from rendergt in MeshLab and apply a 'Per Face Color Function' (Filters->Color Creation and Processing->Per Face Color Function) with the following values:

* func r = 255
* func g = 255-255*wtu0
* func b = 255-255*wtu0
* func alpha = 255

You must also set the Face Color to 'Face', not 'Mesh' or 'User-Def'.

## Example Data Loader
An example data loader for the dataset is provided in [python/exampleDataLoader.py](./python/exampleDataLoader.py). The script loads poses and depth frames from a C3VD sequence and reprojects them into a 3D point cloud.

## Reference
If you find our work useful in your research, please consider citing our paper:
```
@article{bobrow2023,
    title={Colonoscopy 3D video dataset with paired depth from 2D-3D registration},
    author={Bobrow, Taylor L and Golhar, Mayank and Vijayan, Rohan and Akshintala, Venkata S and Garcia, Juan R and Durr, Nicholas J},
    journal={Medical Image Analysis},
    pages={102956},
    year={2023},
    publisher={Elsevier}
}
```

## License
This work is licensed under CC BY-NC-SA 4.0
