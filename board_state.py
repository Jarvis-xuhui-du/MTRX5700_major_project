import cv2
import numpy as np
import pyrealsense2 as rs

# Parameters

BOARD_SIZE = 9
WARP_SIZE = 700

EMPTY = 0
HUMAN_GREEN = 1
ROBOT_RED = 2

TOP_LEFT_ID = 0
TOP_RIGHT_ID = 1
BOTTOM_RIGHT_ID = 2
BOTTOM_LEFT_ID = 3

STABLE_REQUIRED = 1

MOTION_THRESHOLD = 18
MOTION_PIXELS_THRESHOLD = 5000
EXPECTED_MIN_GRID_EDGES = 9000

DEPTH_RADIUS = 30

# Tight ROI for spare pieces only
SPARE_ROI = (455, 210, 535, 500)

# Global state

previous_warped_gray = None
previous_logged_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Camera view

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    print("\nCamera intrinsics:")
    print("width:", intrinsics.width)
    print("height:", intrinsics.height)
    print("fx:", intrinsics.fx)
    print("fy:", intrinsics.fy)
    print("cx:", intrinsics.ppx)
    print("cy:", intrinsics.ppy)

    return pipeline, align, intrinsics


def get_frame(pipeline, align):
    frames = pipeline.wait_for_frames(timeout_ms=10000)
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None

    frame = np.asanyarray(color_frame.get_data())

    return frame, depth_frame


# Board detection

def detect_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        detector_params
    )

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    ids = ids.flatten()

    marker_dict = {}

    for marker_corners, marker_id in zip(corners, ids):
        marker_dict[int(marker_id)] = marker_corners[0]

    required_ids = [
        TOP_LEFT_ID,
        TOP_RIGHT_ID,
        BOTTOM_RIGHT_ID,
        BOTTOM_LEFT_ID
    ]

    for rid in required_ids:
        if rid not in marker_dict:
            return None

    top_left = marker_dict[TOP_LEFT_ID][2]
    top_right = marker_dict[TOP_RIGHT_ID][3]
    bottom_right = marker_dict[BOTTOM_RIGHT_ID][0]
    bottom_left = marker_dict[BOTTOM_LEFT_ID][1]

    board_corners = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    return board_corners


# Wrap board
def warp_board(frame, board_corners):
    dst = np.array([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(board_corners, dst)

    warped = cv2.warpPerspective(
        frame,
        H,
        (WARP_SIZE, WARP_SIZE)
    )

    return warped


# Grid detection
def detect_grid_rectangle(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        130,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        if area < 50000:
            continue

        aspect = w / float(h)

        if 0.7 < aspect < 1.3:
            candidates.append((x, y, w, h, area))

    if len(candidates) == 0:
        return None

    x, y, w, h, _ = max(candidates, key=lambda item: item[4])

    pad = 2

    return (
        x + pad,
        y + pad,
        w - 2 * pad,
        h - 2 * pad
    )


def get_grid_points(grid_rect):
    x, y, w, h = grid_rect

    grid_points = []

    for row in range(BOARD_SIZE):
        row_points = []

        for col in range(BOARD_SIZE):
            px = int(x + col * w / (BOARD_SIZE - 1))
            py = int(y + row * h / (BOARD_SIZE - 1))

            row_points.append((px, py))

        grid_points.append(row_points)

    return grid_points


# Hand detection
def detect_hand_on_board(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 15, 60])
    upper_skin = np.array([30, 180, 255])

    skin_mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    kernel = np.ones((7, 7), np.uint8)

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        skin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_mask = np.zeros_like(skin_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ignore gomoku piece
        if area > 7000:
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                -1
            )

    skin_pixels = cv2.countNonZero(filtered_mask)

    HAND_PIXEL_THRESHOLD = 10000

    hand_detected = skin_pixels > HAND_PIXEL_THRESHOLD

    return hand_detected, filtered_mask, skin_pixels



# Board occlusion
def is_board_occluded(warped):
    global previous_warped_gray

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    if previous_warped_gray is None:
        previous_warped_gray = blur.copy()
        return False, 0, 0, 0

    diff = cv2.absdiff(previous_warped_gray, blur)

    _, motion_mask = cv2.threshold(
        diff,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((7, 7), np.uint8)

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    motion_pixels = cv2.countNonZero(motion_mask)

    grid_rect = detect_grid_rectangle(warped)

    blockage_pixels = 0
    edge_pixels = 0

    if grid_rect is not None:
        x, y, w, h = grid_rect

        roi_gray = gray[y:y + h, x:x + w]

        edges = cv2.Canny(roi_gray, 50, 150)

        edge_pixels = cv2.countNonZero(edges)

        if edge_pixels < EXPECTED_MIN_GRID_EDGES:
            blockage_pixels = w * h

    previous_warped_gray = blur.copy()

    motion_occluded = motion_pixels > MOTION_PIXELS_THRESHOLD
    grid_blocked = blockage_pixels > 0

    occluded = motion_occluded or grid_blocked

    total_score = motion_pixels + blockage_pixels

    return (
        occluded,
        motion_pixels,
        edge_pixels,
        total_score
    )



# Gomoku piece defination
def classify_piece(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # GREEN
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Red piece
    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([18, 255, 255])

    lower_red2 = np.array([160, 50, 40])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = red_mask1 + red_mask2

    green_score = cv2.countNonZero(green_mask)
    red_score = cv2.countNonZero(red_mask)

    threshold = 35

    if green_score > threshold and green_score > red_score:
        return HUMAN_GREEN

    if red_score > threshold and red_score > green_score:
        return ROBOT_RED

    return EMPTY



# Board state detection
def detect_board_state(warped):
    board_state = np.zeros(
        (BOARD_SIZE, BOARD_SIZE),
        dtype=int
    )

    grid_rect = detect_grid_rectangle(warped)

    if grid_rect is None:
        return board_state, None

    grid_points = get_grid_points(grid_rect)

    crop_radius = 18

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x, y = grid_points[row][col]

            crop = warped[
                max(0, y - crop_radius):min(WARP_SIZE, y + crop_radius),
                max(0, x - crop_radius):min(WARP_SIZE, x + crop_radius)
            ]

            board_state[row, col] = classify_piece(crop)

    return board_state, grid_rect


# Print new piece state
previous_logged_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def print_new_piece_update(board_state):
    global previous_logged_state

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):

            old_value = previous_logged_state[row, col]
            new_value = board_state[row, col]

            if old_value != new_value:
                print(
                    f"UPDATED INTERSECTION -> "
                    f"row={row}, col={col}, colour={new_value}"
                )

    previous_logged_state = board_state.copy()



# Visualise board state
def draw_board_state_only(
    warped,
    board_state,
    grid_rect,
    hand_detected=False
):
    debug = warped.copy()

    if grid_rect is None:
        return debug

    grid_points = get_grid_points(grid_rect)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            px, py = grid_points[row][col]
            state = board_state[row, col]

            if state == EMPTY:
                colour = (255, 0, 0)
            elif state == HUMAN_GREEN:
                colour = (0, 255, 0)
            else:
                colour = (0, 0, 255)

            cv2.circle(debug, (px, py), 8, colour, 2)

            cv2.putText(
                debug,
                str(state),
                (px - 5, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                2
            )

    if hand_detected:
        cv2.putText(
            debug,
            "HAND DETECTED - HOLDING STATE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return debug



# Main
def main():
    pipeline, align, intrinsics = start_realsense()

    previous_board_state = np.zeros(
        (BOARD_SIZE, BOARD_SIZE),
        dtype=int
    )

    candidate_board_state = None
    stable_count = 0

    last_debug_warped = None

    try:
        print("\nPress q to quit")

        while True:

            frame, depth_frame = get_frame(
                pipeline,
                align
            )

            if frame is None:
                continue

            board_corners = detect_board_corners(frame)

            if board_corners is None:

                if last_debug_warped is not None:
                    cv2.imshow(
                        "Board State",
                        last_debug_warped
                    )

                key = cv2.waitKey(40) & 0xFF

                if key == ord("q"):
                    break

                continue

            warped = warp_board(
                frame,
                board_corners
            )

            hand_detected, skin_mask, skin_pixels = detect_hand_on_board(
                warped
            )

            cv2.imshow(
                "Skin Mask",
                skin_mask
            )

            occluded, motion_pixels, edge_pixels, total_score = is_board_occluded(
                warped
            )

            grid_rect = detect_grid_rectangle(warped)

            if occluded or hand_detected:

                board_state = previous_board_state.copy()

                candidate_board_state = None
                stable_count = 0

            else:

                detected_board_state, detected_grid_rect = detect_board_state(
                    warped
                )

                if detected_grid_rect is not None:
                    grid_rect = detected_grid_rect

                if candidate_board_state is None:

                    candidate_board_state = detected_board_state.copy()
                    stable_count = 1

                elif np.array_equal(
                    candidate_board_state,
                    detected_board_state
                ):

                    stable_count += 1

                else:

                    candidate_board_state = detected_board_state.copy()
                    stable_count = 1

                if stable_count >= STABLE_REQUIRED:
                    previous_board_state = detected_board_state.copy()

                board_state = previous_board_state.copy()

            debug_warped = draw_board_state_only(
                warped,
                board_state,
                grid_rect,
                hand_detected=hand_detected
            )

            last_debug_warped = debug_warped.copy()

            print("\nBoard state:")
            print(board_state)

            print_new_piece_update(board_state)

            cv2.imshow(
                "Board State",
                debug_warped
            )

            key = cv2.waitKey(40) & 0xFF

            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()