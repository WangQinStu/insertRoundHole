# Repository Guidelines

## Project Structure & Module Organization
- Core entry point: `main.py` ties together camera input, circle detection, 3D extraction, and live visualization.
- Camera utilities live in `realsense/rs_camera.py`; saved point clouds land in `realsense/data/`.
- Detection logic is in `detection/` (stable `CircleDetector`, AprilTag pose in `QRCodeDetector.py`, tuning scripts `test_detector1.py` / `test_detector2.py`).
- Point-cloud extraction and viewers are in `pose/` (`circle_point_extractor.py`, `circle_point_live_viewer.py`).
- Experiments and notebooks sit in `learning/`; keep production code out of this folder.
- `app/` is reserved for runnable scripts; add new entrypoints here to avoid cluttering the root.

## Environment, Build, and Run
- Python 3.10+ is expected. Create a venv before installing packages.
- Install dependencies (OpenCV with ArUco, RealSense, Open3D, NumPy): `pip install opencv-contrib-python pyrealsense2 open3d numpy`.
- Run the end-to-end demo (requires an attached RealSense camera): `python main.py`.
- Optional viewers/tuners:
  - `python realsense/rs_camera.py` to visualize RGB/depth/point-cloud streams.
  - `python detection/test_detector1.py` or `python detection/test_detector2.py` to tune circle detection thresholds live.
  - `python detection/QRCodeDetector.py` to estimate AprilTag pose and rod tip position.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions/variables, CamelCase for classes.
- Prefer explicit numpy array shapes and type hints where practical (see `RealSenseCamera`).
- Keep OpenCV constants/configuration grouped at the top of modules; mirror existing naming (e.g., `canny_low`, `circularity_thresh`).
- Log concise, actionable messages; keep user-facing strings bilingual only when necessary.

## Testing Guidelines
- No automated test suite exists; rely on the live scripts above. Validate changes against real camera input whenever possible.
- When changing detection logic, record observed FPS and detection stability; keep before/after screenshots of the `Circle Detection 2D` window.
- For pose changes, log the printed 3D coordinates and ensure point-cloud overlays update smoothly without drift.

## Commit & Pull Request Guidelines
- Commit messages follow the existing style: short, imperative summaries (`optimized qrcode detector`).
- Include scope in the summary (e.g., `pose: smooth center update`) and keep the body focused on motivation and impact.
- PRs should state the scenario tested (hardware used, scripts run), list commands executed, and attach representative screenshots or console logs.
- Link related issues/tasks when available and mention any new runtime requirements (driver versions, camera resolution changes).
