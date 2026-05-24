# Gomoku Board Vision System

A real-time computer vision system that uses an Intel RealSense depth camera to detect and track the state of a physical Gomoku (Five in a Row) board. It identifies piece colours, detects hand interference, and reports board changes as they happen.

---

## Features

- **ArUco marker-based board detection** — four corner markers define the board boundary and enable automatic perspective correction
- **Real-time piece classification** — distinguishes between human (green) and robot (red) pieces using HSV colour segmentation
- **Hand / occlusion detection** — pauses state updates when a hand or other obstruction is present over the board
- **Motion-based stability check** — only commits a new board state after it has been observed consistently, avoiding false reads mid-move
- **Live debug visualisation** — annotated warped board view showing piece states and detected intersections

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Intel RealSense camera | Colour + depth streams at 640×480, 15 fps |
| Physical Gomoku board | 9×9 grid |
| ArUco markers (DICT_4X4_50) | IDs 0–3 placed at the four board corners |
| Coloured pieces | Green (human) and Red (robot) |

---

## Software Requirements

```
Python 3.8+
opencv-python
opencv-contrib-python   # required for ArUco support
numpy
pyrealsense2
```

Install dependencies:

```bash
pip install opencv-python opencv-contrib-python numpy pyrealsense2
```

---

## ArUco Marker Placement

Place the four markers around the board corners in the following positions:

```
ID 0 (TOP_LEFT)      ID 1 (TOP_RIGHT)
        ┌─────────────────┐
        │                 │
        │   Gomoku Board  │
        │                 │
        └─────────────────┘
ID 3 (BOTTOM_LEFT)   ID 2 (BOTTOM_RIGHT)
```

The system uses a specific corner of each marker as the board reference point, so ensure markers are flat and unobstructed.

---

## Usage

```bash
python solution.py
```

- The board state is printed to the console each frame.
- Any intersection that changes value is logged immediately with its row, column, and new colour.
- Press **`q`** to quit.

---

## Configuration

Key parameters at the top of the file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOARD_SIZE` | `9` | Number of intersections per side |
| `WARP_SIZE` | `700` | Output size (px) of the perspective-corrected board |
| `STABLE_REQUIRED` | `1` | Consecutive matching frames before a state is committed |
| `MOTION_THRESHOLD` | `18` | Per-pixel difference to count as motion |
| `MOTION_PIXELS_THRESHOLD` | `5000` | Total motion pixels to trigger occlusion |
| `EXPECTED_MIN_GRID_EDGES` | `9000` | Minimum Canny edge pixels expected inside the grid |
| `DEPTH_RADIUS` | `30` | Radius (px) used for depth sampling (future use) |
| `SPARE_ROI` | `(455,210,535,500)` | ROI for spare piece area (future use) |

---

## Board State Values

| Value | Constant | Meaning |
|-------|----------|---------|
| `0` | `EMPTY` | No piece at this intersection |
| `1` | `HUMAN_GREEN` | Green piece placed by the human player |
| `2` | `ROBOT_RED` | Red piece placed by the robot |

---

## System Pipeline

```
RealSense Frame (colour + depth)
        │
        ▼
Detect ArUco Corners  ──── not found ──▶  show last frame
        │
        ▼
Perspective Warp (700×700)
        │
        ├──▶ Hand Detection (HSV skin mask)
        │
        ├──▶ Occlusion Check (frame diff + edge count)
        │
        ▼
  Occluded / Hand?
   YES ──▶ Hold previous board state
   NO  ──▶ Detect Grid → Classify each intersection
                │
                ▼
         Stability Check
                │
                ▼
         Commit Board State → Print changes → Visualise
```

---

## Output

Console output on each state change:

```
Camera intrinsics
9x9 matrix board state
Updated intersection -> row=4, col=4, colour=1
```

Two OpenCV windows are displayed:

| Window | Contents |
|--------|----------|
| `Board State` | Warped board with colour-coded intersection labels |
| `Skin Mask` | Binary mask used for hand detection |

---

## Known Limitations

- Piece classification relies on HSV colour thresholds — strong or inconsistent lighting may require tuning 'MOTION_PIXELS_THRESHOLD = 5000', 'EXPECTED_MIN_GRID_EDGES' = 9000, 'HAND_PIXEL_THRESHOLD', 'area > 7000'. 
- `STABLE_REQUIRED = 1` means a single clean frame commits the state; increase this value if noisy readings are a problem.
