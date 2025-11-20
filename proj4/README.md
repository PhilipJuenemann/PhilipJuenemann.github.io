# Project 4 - Part 0: Camera Calibration and NeRF Dataset Creation

## Overview

This project implements a complete pipeline for camera calibration and NeRF dataset creation using ArUco markers.

## Implemented Parts

### Part 0.1: Camera Calibration ✅
- Calibrates camera using 30-50 images of ArUco tags from multiple angles
- Uses OpenCV's `cv2.calibrateCamera()` to compute intrinsic parameters
- **Important**: Uses 55mm tag size (not 60mm) as tags were printed at that size
- Outputs: Camera matrix (K) and distortion coefficients
- Results saved to: `camera_calibration.npz`

### Part 0.2: 3D Object Scan Capture ✅
- Captured 30-50 images of an object with a single ArUco tag
- Images stored in: `single_aruco_images/` folder
- Same camera and zoom level as calibration

### Part 0.3: Camera Pose Estimation ✅
- Detects single ArUco tag in each image
- Uses `cv2.solvePnP()` to estimate camera pose (rotation + translation)
- Converts world-to-camera (w2c) to camera-to-world (c2w) transformation matrices
- **Important**: Uses 55mm tag size for pose estimation
- Visualizes camera frustums in 3D using Viser
- Results saved to: `camera_poses.npz`

### Part 0.4: Undistortion and Dataset Creation ✅
- Undistorts all images using `cv2.undistort()`
- Removes black borders using `cv2.getOptimalNewCameraMatrix()`
- Updates camera intrinsics to account for cropping
- Splits data into train/val/test sets (70%/15%/15%)
- Saves in NeRF-compatible format
- Output: `my_data.npz` (or custom name)

## Usage

Run the main script:
```bash
python main.py
```

The script will interactively guide you through:
1. Camera calibration (or load existing)
2. Pose estimation
3. Dataset creation with configurable options
4. 3D visualization with Viser

## Dataset Format

The output `.npz` file contains:
- `images_train`: (N_train, H, W, 3) uint8 array in 0-255 range
- `c2ws_train`: (N_train, 4, 4) float32 camera-to-world matrices
- `images_val`: (N_val, H, W, 3) uint8 validation images
- `c2ws_val`: (N_val, 4, 4) float32 validation poses
- `c2ws_test`: (N_test, 4, 4) float32 test poses (for novel views)
- `focal`: float focal length in pixels
- `camera_matrix`: (3, 3) updated camera intrinsics (for reference)
- `image_size`: (H, W) final image dimensions (for reference)

## Key Implementation Details

### ArUco Tag Size
The physical ArUco tags are **55mm**, not 60mm. This is clearly marked in the code:
- Line 50 in `calibrate_camera()`: `tag_size = 0.055  # 55mm in meters`
- Line 203 in `estimate_camera_poses()`: `tag_size = 0.055  # 55mm in meters`

### Black Border Removal
The script uses `cv2.getOptimalNewCameraMatrix()` with configurable alpha:
- `alpha=0`: Maximum cropping (removes all black borders)
- `alpha=1`: Keeps all pixels (more black borders)
- Updates principal point (cx, cy) to account for crop offset

### Coordinate Systems
- World origin: ArUco tag position (z=0 plane)
- OpenCV returns w2c, we convert to c2w for NeRF/visualization
- Camera frustums show actual camera positions/orientations

## Functions

- `calibrate_camera()`: Calibrates camera from ArUco images
- `estimate_camera_poses()`: Estimates poses using PnP
- `create_nerf_dataset()`: Undistorts images and creates dataset
- `verify_nerf_dataset()`: Validates dataset file
- `visualize_camera_poses()`: 3D visualization with Viser

## Requirements

```bash
pip install opencv-python numpy viser
```

## Files Generated

- `camera_calibration.npz`: Camera intrinsics and distortion coefficients
- `camera_poses.npz`: Estimated camera poses (without images)
- `my_data.npz`: Complete NeRF dataset with undistorted images

## Deliverables

Part 0.3 deliverables (completed):
- `deliverables/part0.2_viser_screenshot_1_side_view.png`
- `deliverables/part0.3_viser_screenshot_2_top_view.png`

## Notes

- Phone cameras work better than DSLRs (less distortion)
- Keep same zoom level for all captures
- White border around printed ArUco tags is essential for detection
- Dataset can be used directly for NeRF training in Parts 1 and 2

