================================================================================
FEATURE MATCHING FOR AUTOSTITCHING - PROJECT 3B
================================================================================

This project implements automatic image stitching using feature detection,
matching, and RANSAC-based homography estimation. The implementation follows
the paper "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown
et al. with several simplifications.

================================================================================
PROJECT OVERVIEW
================================================================================

The automatic panorama stitching pipeline consists of four main steps:

B.1: Harris Corner Detection & Adaptive Non-Maximal Suppression (ANMS)
B.2: Feature Descriptor Extraction
B.3: Feature Matching using Lowe's Ratio Test
B.4: RANSAC for Robust Homography Estimation

================================================================================
IMPLEMENTATION DETAILS
================================================================================

B.1: HARRIS CORNER DETECTION & ANMS (20 pts)
────────────────────────────────────────────
- Harris Interest Point Detector detects corners in grayscale images
- Uses edge_discard=20 to avoid corners too close to image boundaries
- Adaptive Non-Maximal Suppression (ANMS) selects well-distributed features
- ANMS ensures features are spread across the image, not clustered
- Parameters:
  * topN: Number of strongest Harris corners to consider (e.g., 5000-10000)
  * n_points: Number of ANMS features to keep (e.g., 800-1000)
  * c_robust: Robustness parameter for ANMS (typically 0.8-0.9)

Key Functions:
- get_harris_corners(): Detects Harris corners using scikit-image
- anms_nearest_stronger(): Implements ANMS algorithm
  * For each corner, finds distance to nearest stronger corner
  * Keeps corners with largest suppression radii

Deliverables:
✓ Harris corners overlaid on images (before ANMS) - shown as green dots
✓ ANMS-selected corners overlaid on images - shown as red dots


B.2: FEATURE DESCRIPTOR EXTRACTION (20 pts)
────────────────────────────────────────────
- Extract 40x40 patches centered on each feature point
- Downsample to 8x8 using bilinear interpolation
- Normalize to mean=0, std=1 (bias/gain normalization)
- Results in 64-dimensional feature descriptors
- No rotation invariance (axis-aligned patches only)

Key Functions:
- extract_descriptors(): Complete pipeline for one image
  * Detects Harris corners
  * Applies ANMS
  * Extracts and normalizes 40x40 patches
  * Downsamples to 8x8
  * Returns normalized 64-D descriptors

- normalize_patch(): Bias/gain normalization
  * Subtracts mean and divides by standard deviation

Deliverables:
✓ Normalized 8x8 feature descriptors extracted for all feature points
✓ Visualization available in commented code (patches vs downsampled)


B.3: FEATURE MATCHING (20 pts)
────────────────────────────────────────────
- Match features between image pairs using Euclidean distance
- Apply Lowe's ratio test: d1/d2 < tau (tau ≈ 0.75-0.80)
  * d1: distance to best match
  * d2: distance to second-best match
- Cross-check: Only keep mutual best matches
- Filters out ambiguous matches

Key Functions:
- pairwise_euclidean(): Computes all pairwise distances efficiently
- match_descriptors_ratio(): Implements Lowe's ratio test + cross-check
  * Finds two nearest neighbors for each descriptor
  * Keeps matches where ratio < tau
  * Optionally enforces mutual matching

- draw_matches(): Visualizes feature correspondences
  * Shows images side-by-side with lines connecting matches

Parameters:
- tau: Ratio threshold (0.75-0.80 typical)
- cross_check: Enforce mutual best matches (True recommended)

Deliverables:
✓ Matched features shown with lines between image pairs
✓ Separate visualizations before and after RANSAC filtering


B.4: RANSAC FOR ROBUST HOMOGRAPHY (40 pts)
────────────────────────────────────────────
- 4-point RANSAC to estimate homography robustly
- Iteratively samples 4 random correspondences
- Computes homography using least-squares (from Part A)
- Counts inliers (reprojection error < threshold)
- Refines homography using all inliers

Key Functions:
- ransac_homography(): Main RANSAC implementation
  * Samples 4 random correspondences per iteration
  * Computes candidate homography using computeH()
  * Evaluates inliers based on reprojection error
  * Keeps best hypothesis
  * Refines on all inliers

- reprojection_errors(): Computes pixel-level errors
- to_homogeneous(): Converts 2D points to homogeneous coordinates
- project_points(): Projects points through homography

- stitch_three_images(): Complete automatic stitching pipeline
  * Detects features in all three images
  * Matches LEFT→CENTER and RIGHT→CENTER
  * Runs RANSAC for both pairs
  * Warps and blends images into panorama
  * Reuses blending code from Part A

Parameters:
- thresh: RANSAC inlier threshold in pixels (typically 3.0)
- max_iters: Number of RANSAC iterations (1000-3000)

Deliverables:
✓ 4-point RANSAC implemented from scratch
✓ Visualizations of RANSAC inliers (after filtering outliers)
✓ Side-by-side comparison of manual vs automatic stitching
✓ Multiple automatic panoramas (tower, field, street)

================================================================================
HOW TO RUN THE CODE
================================================================================

PREREQUISITES:
--------------
- Python 3.7+
- Required packages:
  * numpy
  * opencv-python (cv2)
  * matplotlib
  * scikit-image
  * scipy (if needed for dependencies)

Install with:
  pip install numpy opencv-python matplotlib scikit-image scipy


DIRECTORY STRUCTURE:
--------------------
proj3B/
  ├── main.py                  # Main implementation file
  ├── README.txt              # This file
  └── images/                 # Output images saved here

proj3A/
  ├── point_alignment.py      # Contains computeH() and blending functions
  └── images/
      ├── tower_left.png
      ├── tower_center.png
      ├── tower_right.png
      ├── field_left.png
      ├── field_center.png
      ├── field_right.png
      ├── street_left.png
      ├── street_center.png
      └── street_right.png


RUNNING THE CODE:
-----------------
1. Navigate to proj3B directory:
   cd proj3B

2. Open main.py and set the image_name variable (line 13):
   image_name = "tower"    # or "field" or "street"

3. Run the script:
   python main.py

4. The script will:
   - Display Harris corners (green dots, many thousands)
   - Display ANMS-selected corners (red dots, ~800-1000)
   - Display feature matches RIGHT→CENTER (before RANSAC)
   - Display RANSAC inliers RIGHT→CENTER (after filtering)
   - Display feature matches LEFT→CENTER (before RANSAC)
   - Display RANSAC inliers LEFT→CENTER (after filtering)
   - Display final stitched panorama
   - Print statistics at each step


CUSTOMIZING PARAMETERS:
-----------------------
At the bottom of main.py (around line 710), modify:

results = stitch_three_images(
    left_path=f"../proj3A/images/{image_name}_left.png",
    center_path=f"../proj3A/images/{image_name}_center.png",
    right_path=f"../proj3A/images/{image_name}_right.png",
    n_points=800,           # Number of ANMS features
    topN=5000,              # Harris corners to consider
    c_robust=0.8,           # ANMS robustness
    tau=0.80,               # Lowe's ratio threshold
    ransac_thresh=3.0,      # RANSAC inlier threshold (pixels)
    max_iters=1000,         # RANSAC iterations
    blend_power=2,          # Blending distance weight
    seed=42,                # Random seed for reproducibility
    show_figs=True          # Display visualizations
)

Tips:
- Increase n_points for more matches (but slower)
- Increase max_iters for better RANSAC convergence
- Decrease tau for stricter feature matching
- Increase ransac_thresh for more lenient inliers


SAVING OUTPUTS:
---------------
To save visualizations, add after plt.show() calls:
  plt.savefig('images/output_name.png', dpi=150, bbox_inches='tight')

To save the final panorama programmatically:
  cv2.imwrite('images/panorama.jpg', panorama)


================================================================================
ALGORITHM WORKFLOW
================================================================================

For each image (LEFT, CENTER, RIGHT):
  1. Detect Harris corners (~10,000+ corners)
  2. Apply ANMS to select ~800-1000 well-distributed points
  3. Extract 40x40 patches around each point
  4. Downsample to 8x8 and normalize (64-D descriptors)

For each pair (LEFT→CENTER, RIGHT→CENTER):
  5. Match descriptors using Lowe's ratio test + cross-check
  6. Run 4-point RANSAC to find robust homography
     - Sample 4 random correspondences
     - Compute homography
     - Count inliers (reprojection error < 3px)
     - Repeat 1000-3000 times
     - Refine using all inliers
  7. Warp and blend images using homography

Final output:
  8. Create three-image panorama (LEFT + CENTER + RIGHT)


================================================================================
KEY DIFFERENCES FROM MANUAL STITCHING (PART A)
================================================================================

PART A (Manual):
- User manually clicks corresponding points
- Requires 4+ point correspondences
- Direct homography computation
- Time-consuming, error-prone

PART B (Automatic):
- Fully automatic feature detection
- Hundreds of feature matches found
- RANSAC filters outliers automatically
- Fast, repeatable, robust


================================================================================
RESULTS & OBSERVATIONS
================================================================================

Typical Performance:
- Harris corners detected: 5,000-20,000 per image
- ANMS features selected: 800-1,000 per image
- Feature matches found: 200-400 per pair
- RANSAC inliers: 150-300 per pair (60-80% inlier ratio)

The automatic method produces results comparable to manual stitching but:
✓ Much faster (seconds vs minutes)
✓ More robust to outliers (RANSAC filtering)
✓ Repeatable (same parameters = same result)
✓ Scales to many images


================================================================================
TROUBLESHOOTING
================================================================================

Problem: Too few matches found
Solution:
  - Lower tau (e.g., 0.85)
  - Increase n_points (e.g., 1200)
  - Check image overlap is sufficient

Problem: Poor stitching quality
Solution:
  - Increase max_iters (e.g., 3000)
  - Adjust ransac_thresh (try 2.5 or 4.0)
  - Ensure images have good overlap

Problem: RANSAC fails
Solution:
  - Check that enough matches exist (>20)
  - Increase max_iters
  - Verify images are related (not completely different scenes)

Problem: Visualizations not showing
Solution:
  - Ensure show_figs=True
  - Check matplotlib backend (may need plt.ion())
  - Run in interactive Python environment


================================================================================
CODE STRUCTURE
================================================================================

Core Functions:
1. get_harris_corners()           - Harris corner detection
2. anms_nearest_stronger()        - ANMS feature selection
3. extract_descriptors()          - Complete feature extraction pipeline
4. normalize_patch()              - Descriptor normalization
5. pairwise_euclidean()           - Distance computation
6. match_descriptors_ratio()      - Lowe's ratio test matching
7. ransac_homography()            - 4-point RANSAC
8. reprojection_errors()          - Error computation
9. stitch_three_images()          - Complete stitching pipeline
10. draw_matches()                - Visualization helper

From proj3A (reused):
- computeH()                      - Homography estimation (least-squares)
- simple_bwdist_blend()           - Two-image blending
- three_image_panorama_blend()    - Three-image blending


================================================================================
REFERENCES
================================================================================

[1] Brown, M., & Lowe, D. G. (2007). "Automatic Panoramic Image Stitching
    using Invariant Features". International Journal of Computer Vision.

[2] Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant
    Keypoints". International Journal of Computer Vision.

[3] Harris, C., & Stephens, M. (1988). "A Combined Corner and Edge Detector".
    Alvey Vision Conference.


================================================================================
AUTHOR & ACKNOWLEDGMENTS
================================================================================

Implementation for CS180/CS280A - Computational Photography
UC Berkeley, Fall 2024

Key algorithms based on:
- Brown et al. Multi-Image Matching paper
- Harris corner detection
- Lowe's ratio test for feature matching
- RANSAC for robust estimation

================================================================================
