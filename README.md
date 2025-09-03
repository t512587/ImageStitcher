# ImageStitcher — Dual‑Camera Shelf Stitching

Create a single, wide “whole shelf” image from two top‑down cameras. The cameras are mounted above the shelf with similar height and intrinsics, one angled slightly left and the other slightly right to cover the full width. This project detects features, estimates a homography, warps the views, blends the overlap, and trims black borders.

This repository is a streamlined and optimized fork tailored for shelf imaging with similar camera poses.

![image](https://github.com/user-attachments/assets/f2ff20b1-54c3-4182-a8be-4c0301986d42)
![image](https://github.com/user-attachments/assets/269bc127-dbc9-4efd-82f2-ac661c74858b)
![image](https://github.com/user-attachments/assets/00962c0e-3e86-4b9b-9bc6-58c986186327)

## Features
- **SIFT feature detection/description** (OpenCV) and **BFMatcher + Lowe’s ratio** filtering
- **RANSAC homography** estimation for robust alignment
- **Warp modes**:
  - **center**: both images are warped toward a midway view (reduces perspective bias)
  - **right_to_left**: warp right image into left image’s frame (classic panorama)
- **Blending modes**:
  - **noBlending**: simple overwrite
  - **linearBlending**: feathering across the overlap
  - **linearBlendingWithConstant**: feathering only near the seam to minimize ghosting
- **Automatic black border trimming**
- **Visualizations**: matching points figure and overlap mask (for debugging)


## Requirements
- Python 3.9–3.13 (tested with 3.13)
- Packages:
  - `opencv-python>=4.5` (if not, try `opencv-contrib-python`)
  - `numpy`
  - `matplotlib`

On headless or CI environments, prefer `opencv-contrib-python-headless` instead of the GUI build.

Install quickly:
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -U pip
pip install opencv-contrib-python numpy matplotlib
# or: pip install opencv-contrib-python-headless numpy matplotlib
```


## Project layout
```
ImageStitcher/
  img/                 # put your input images here (e.g., 3.jpg, 4.jpg)
  main.py              # simple runner / example usage
  Stitcher.py          # pipeline: detect → match → homography → warp → blend
  Blender.py           # blending strategies
  Homography.py        # (kept for reference) SVD-based homography solver
  README.md
```


## Prepare your inputs (shelf cameras)
- Mount both cameras at similar height and orientation, pointing downward to the shelf.
- Angle one slightly left and the other slightly right so their fields of view overlap by ~20–40%.
- Fix exposure/white balance to minimize brightness differences in the overlap.
- Keep lenses/focal lengths the same if possible.
- Avoid motion; if capturing dynamic scenes, trigger both cameras near-simultaneously.

Place your two images in `img/`, e.g. `img/3.jpg` and `img/4.jpg`.


## Run
```bash
python main.py
```
This will:
- read the two images defined in `main.py`
- run the stitch with your chosen warp/blending modes
- open matplotlib windows to visualize matches and the final stitched output

To save output, you can add in `main.py` after stitching:
```python
import cv2
cv2.imwrite("stitch_3_4.jpg", stitch_img)
```


## Configure
Edit `main.py`:
```python
fileNameList = [("3", "4")]  # change to your filenames (without extension)
blending_mode = "linearBlending"  # "noBlending", "linearBlending", "linearBlendingWithConstant"
stitch_img = stitcher.stitch([img_left, img_right], blending_mode, view_mode="center")
# view_mode: "center" or "right_to_left"
```

Pipeline parameters (in `Stitcher.py`):
- `Stitcher.stitch(imgs, blending_mode="linearBlending", ratio=0.75, view_mode="center")`
  - **ratio**: Lowe’s ratio for feature filtering (lower = stricter, fewer matches; higher = more matches but possibly noisier)
  - **view_mode**: choose `center` for symmetric warping (often better for similar poses), or `right_to_left` for classic panorama
  - **blending_mode**: choose a blending strategy; for shelves, try `linearBlendingWithConstant` first to limit ghosting

Blending details (in `Blender.py`):
- `linearBlendingWithConstantWidth` uses a seam half‑width controlled by `constant_width` (default 3). Increase if your seam is wider or textures are noisy.


## Recommended settings for shelf stitching
- Start with `view_mode="center"` to reduce perspective bias when cameras are similarly placed.
- Use `blending_mode="linearBlendingWithConstant"` to keep feathering near the seam and avoid ghosting of repeated patterns (e.g., labels or edges).
- Ensure sufficient overlap and texture in the overlap (text/edges) so SIFT can find robust matches.
- If matches are too few or alignment is unstable, adjust `ratio` (e.g., 0.6–0.8) and ensure input images have sharp, non-blurry details.


## Troubleshooting
- "module 'cv2' has no attribute 'SIFT_create'": install `opencv-contrib-python` (not `opencv-python`).
- No/poor stitching: verify adequate overlap; check terminal log for "The number of matching points" and inliers; ensure non-blurry inputs.
- Distortion or skew: try `view_mode="center"`; make sure cameras are at similar heights/angles.
- GUI issues on servers: use `opencv-contrib-python-headless` and consider saving figures instead of showing them.


## Programmatic usage
```python
import cv2
from Stitcher import Stitcher

img_left = cv2.imread("img/3.jpg")
img_right = cv2.imread("img/4.jpg")

stitcher = Stitcher()
stitched = stitcher.stitch([img_left, img_right],
                           blending_mode="linearBlendingWithConstant",
                           view_mode="center",
                           ratio=0.75)
cv2.imwrite("shelf_stitched.jpg", stitched)
```


## Notes
- This fork is adapted for the dual top‑down shelf camera scenario with similar camera poses. If you work with very different viewpoints or focal lengths, additional calibration or exposure compensation may be required.
- Sourced from [Automatic Panoramic Image Stitching by Yunyung](https://github.com/Yunyung/Automatic-Panoramic-Image-Stitching)
