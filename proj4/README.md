# Project 4: Neural Radiance Fields (NeRF)

This project implements a complete pipeline for creating, training, and rendering Neural Radiance Fields (NeRF) from scratch. It covers everything from camera calibration and dataset creation to training a NeRF model and synthesizing novel views.

## Project Structure

The project is organized into three main parts:

1.  **Part 0: Camera Calibration and Dataset Creation**: Capturing real-world data and preparing it for NeRF.
2.  **Part 1: 2D Neural Fields**: Learning to represent a single 2D image as a neural field.
3.  **Part 2: 3D NeRF**: Training a NeRF on multi-view images (synthetic and custom).

## Part 0: Camera Calibration and Dataset Creation

**Goal**: Create a custom dataset of a real object for NeRF training.

### Steps:
1.  **Camera Calibration (`part0.1`)**:
    - Captured 36 images of ArUco markers.
    - Used `cv2.calibrateCamera` to compute intrinsic parameters (focal length, principal point, distortion coefficients).
    - **Note**: Physical tag size was 55mm.

2.  **Pose Estimation (`part0.3`)**:
    - Captured 50 images of a "Lafufu" object with a single ArUco tag.
    - Used `cv2.solvePnP` to estimate the camera pose relative to the tag.
    - Converted poses to camera-to-world (c2w) matrices.
    - Visualized camera frustums using `viser` to verify coverage.

3.  **Dataset Formatting (`part0.4`)**:
    - Undistorted images to remove lens distortion (NeRF assumes pinhole camera).
    - Cropped images to remove black borders.
    - Saved data in a `.npz` file containing:
        - `images_train/val/test`: RGB images.
        - `c2ws_train/val/test`: Camera poses.
        - `K`: Intrinsic matrix.

**Key Scripts**:
- `main.py`: Interactive script guiding through calibration, pose estimation, and dataset creation.

## Part 1: Fit a Neural Field to a 2D Image

**Goal**: Train a Multi-Layer Perceptron (MLP) to map 2D pixel coordinates `(x, y)` to RGB colors.

### Implementation:
- **Architecture**: 4-layer MLP with 256 hidden units.
- **Positional Encoding**: Applied sinusoidal encoding (L=10 frequencies) to inputs `(x, y)` to enable learning high-frequency details (texture, edges).
- **Training**:
    - Sampled random pixels from the image.
    - Optimized Mean Squared Error (MSE) loss using Adam optimizer.
    - Achieved ~30 dB PSNR on the "fox" image.

**Key Findings**:
- Without positional encoding, the network produces blurry, low-frequency approximations.
- Increasing network width and encoding frequencies improves detail reconstruction.

**Key Scripts**:
- `part1_neural_field.py`: Training script for 2D neural fields.

## Part 2: Fit a Neural Radiance Field from Multi-view Images

**Goal**: Train a 3D NeRF to synthesize novel views of a scene given a set of input images and camera poses.

### Implementation Details:
1.  **Ray Sampling**:
    - Converted pixel coordinates to rays using camera intrinsics and c2w matrices.
    - Implemented `pixel_to_ray`, `pixel_to_camera`, and `transform` functions.

2.  **Neural Network (`NeRFMLP`)**:
    - Input: 3D position `(x, y, z)` and viewing direction `(theta, phi)`.
    - Output: Volume density `sigma` and RGB color.
    - Architecture: 8-layer MLP (width 256) with skip connections.
    - Directional input is injected late in the network to model view-dependent effects (specularities).

3.  **Volume Rendering**:
    - Implemented the differentiable volume rendering equation.
    - Sampled 64 points along each ray (`near=2.0`, `far=6.0` for synthetic data).
    - Accumulated color and opacity along the ray.

### Results:
- **Lego Dataset (Synthetic)**:
    - Trained for 2000 iterations.
    - Achieved validation PSNR > 26 dB.
    - Rendered a 360° spiral video demonstrating stable geometry and view-dependent effects.

- **Lafufu Dataset (Provided Real Data)**:
    - Used provided calibrated data for validation.
    - Adjusted hyperparameters: `near=0.02`, `far=0.5` (smaller scale).
    - Successfully reconstructed the object geometry.

- **Custom Dataset (My Capture)**:
    - Trained on the dataset created in Part 0.
    - **Challenges**: Real-world data is messier (lighting variations, pose errors).
    - **Adjustments**:
        - Increased iterations to 5000.
        - Resized images to 256px width to fit in GPU memory.
        - Used `near=0.02`, `far=0.5`.
    - **Outcome**: The model learned the general structure and color of the object, though with more artifacts than the synthetic datasets, likely due to small inaccuracies in the estimated poses or lighting changes during capture.

**Key Scripts**:
- `part2_nerf.py`: Core NeRF implementation (model, rendering, training loop).
- `run_lafufu.py`: Specialized training script for the Lafufu dataset.
- `run_custom_nerf.py`: Training script tailored for the custom dataset.
- `render_custom_video.py`: Script to render novel view videos (spirals/spins) from trained models.

## How to Run

### Dependencies
```bash
pip install torch numpy opencv-python imageio tqdm matplotlib viser
```

### 1. 2D Neural Field (Part 1)
```bash
python part1_neural_field.py --image images/fox.jpg --iter 2000
```

### 2. Train NeRF on Lego (Part 2)
```bash
python part2_nerf.py
```

### 3. Train NeRF on Lafufu
```bash
python proj4/run_lafufu.py
```

### 4. Train NeRF on Custom Dataset
First, ensure your `camera_poses.npz` is in `datasets/`.
```bash
python proj4/run_custom_nerf.py
```

### 5. Render Video from Trained Model
To generate a video (e.g., horizontal spin) from a saved model checkpoint:
```bash
python proj4/render_custom_video.py
```

## Visualizations

See `index.html` for a comprehensive report including:
- Training progression grids.
- PSNR curves.
- Final rendered gifs for Fox, Lego, Lafufu, and Custom datasets.
