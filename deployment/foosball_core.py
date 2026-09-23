"""
Core foosball video processing pipeline.

Pipeline: video path -> tracking DataFrame -> 6 feature sequences
-> TCN model -> per-sequence predictions -> hard/soft vote -> winner
"""
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Physical table configuration

FIELD_WIDTH_MM  = 1200
FIELD_HEIGHT_MM = 680
PAD             = 50
H_PLATFORM      = -118.0
H_ROD           = -85.0
BORDER_WIDTH    = 9.5

HIT_ZONE_WIDTH  = 120
HIT_ZONE_HEIGHT = 80
GOAL_Y_CENTER   = FIELD_HEIGHT_MM / 2
GOAL_TOLERANCE  = 100

LOWER_RED = np.array([152, 101, 175], dtype=np.uint8)
UPPER_RED = np.array([175, 255, 255], dtype=np.uint8)

BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 100
MIN_MOVEMENT_MM       = 3

STOPPED_THRESHOLD    = 0.10
SLOW_EXIT_THRESHOLD  = 0.30
FAST_EXIT_THRESHOLD  = 0.50
MIN_DIRECTION_CHANGE = 25
BLOCK_ANGLE          = 150
MIN_FRAMES_CONTROL   = 3

ROD_CONFIG = {
    1: {'x': 75,   'team': 'Black', 'role': 'GK',  'num_players': 1, 'player_color': 'dark',  'offsets_bottom': None, 'offsets_top': None},
    2: {'x': 225,  'team': 'Black', 'role': 'DEF', 'num_players': 2, 'player_color': 'dark',  'offsets_bottom': [15, -219], 'offsets_top': [-15, 219]},
    3: {'x': 375,  'team': 'White', 'role': 'ATK', 'num_players': 3, 'player_color': 'light', 'offsets_bottom': [-15, -199, -383], 'offsets_top': [15, 199, 383]},
    4: {'x': 525,  'team': 'Black', 'role': 'MID', 'num_players': 5, 'player_color': 'dark',  'offsets_bottom': [15, -106.5, -228, -349.5, -471], 'offsets_top': [-15, 106.5, 228, 349.5, 471]},
    5: {'x': 675,  'team': 'White', 'role': 'MID', 'num_players': 5, 'player_color': 'light', 'offsets_bottom': [-15, -136.5, -258, -379.5, -501], 'offsets_top': [15, 136.5, 258, 379.5, 501]},
    6: {'x': 825,  'team': 'Black', 'role': 'ATK', 'num_players': 3, 'player_color': 'dark',  'offsets_bottom': [15, -169, -353], 'offsets_top': [-15, 169, 353]},
    7: {'x': 975,  'team': 'White', 'role': 'DEF', 'num_players': 2, 'player_color': 'light', 'offsets_bottom': [-15, -249], 'offsets_top': [15, 249]},
    8: {'x': 1125, 'team': 'White', 'role': 'GK',  'num_players': 1, 'player_color': 'light', 'offsets_bottom': None, 'offsets_top': None},
}

qr_world_points = np.array([
    [-BORDER_WIDTH, BORDER_WIDTH, H_PLATFORM],
    [FIELD_WIDTH_MM + BORDER_WIDTH, BORDER_WIDTH, H_PLATFORM],
    [FIELD_WIDTH_MM + BORDER_WIDTH, FIELD_HEIGHT_MM - BORDER_WIDTH, H_PLATFORM],
    [-BORDER_WIDTH, FIELD_HEIGHT_MM - BORDER_WIDTH, H_PLATFORM]
], dtype=np.float32)

field_world_points = np.array([
    [0, 0, 0], [FIELD_WIDTH_MM, 0, 0],
    [FIELD_WIDTH_MM, FIELD_HEIGHT_MM, 0], [0, FIELD_HEIGHT_MM, 0]
], dtype=np.float32)

dst_points = np.array([
    [PAD, PAD], [FIELD_WIDTH_MM + PAD, PAD],
    [FIELD_WIDTH_MM + PAD, FIELD_HEIGHT_MM + PAD], [PAD, FIELD_HEIGHT_MM + PAD]
], dtype=np.float32)



# Homography / tracking helpers

def get_marker_data(img):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def compute_homography_and_pose(img_undist, K_new):
    corners, ids = get_marker_data(img_undist)
    if ids is None:
        return None, None, None
    ids_flat = np.array(ids).reshape(-1)
    marker_dict = {int(ids_flat[i]): corners[i].reshape(4, 2) for i in range(len(ids_flat))}
    if not all(m in marker_dict for m in [0, 1, 2, 3]):
        return None, None, None
    qr_image_points = np.array([
        marker_dict[0][2], marker_dict[1][3],
        marker_dict[2][0], marker_dict[3][1]
    ], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(qr_world_points, qr_image_points, K_new, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None, None
    rvec, tvec = cv2.solvePnPRefineLM(qr_world_points, qr_image_points, K_new, None, rvec, tvec)
    field_corners_img, _ = cv2.projectPoints(field_world_points, rvec, tvec, K_new, None)
    H, _ = cv2.findHomography(field_corners_img.reshape(-1, 2), dst_points)
    return H, rvec, tvec


def project_rod_line_to_warped(rod_x, rvec, tvec, K_new, H, num_samples=200):
    y_positions = np.linspace(0, FIELD_HEIGHT_MM, num_samples)
    rod_3d = np.array([[rod_x, y, H_ROD] for y in y_positions], dtype=np.float32)
    rod_img, _ = cv2.projectPoints(rod_3d, rvec, tvec, K_new, None)
    warped_pts = []
    for pt in rod_img.reshape(-1, 2):
        wpt = H @ np.array([pt[0], pt[1], 1.0])
        warped_pts.append((int(wpt[0] / wpt[2]), int(wpt[1] / wpt[2])))
    return warped_pts, y_positions


def extract_intensities_along_line(image, points):
    h, w = image.shape[:2]
    intensities = []
    for (x, y) in points:
        x, y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
        if len(image.shape) == 3:
            bgr = image[y, x]
            intensities.append(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0])
    return np.array(intensities)


def apply_bilateral_filter_1d(intensities):
    signal_2d = intensities.reshape(1, -1).astype(np.float32)
    return cv2.bilateralFilter(signal_2d, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE).flatten()


def get_player_id(rod_num, player_idx):
    return f"{rod_num}.{player_idx + 1}"


def check_ball_in_hit_zone(bx, by, rx, py):
    if bx is None or by is None:
        return False
    return ((rx - HIT_ZONE_WIDTH / 2) <= bx <= (rx + HIT_ZONE_WIDTH / 2) and
            (py - HIT_ZONE_HEIGHT / 2) <= by <= (py + HIT_ZONE_HEIGHT / 2))


def find_ball_in_hit_zones(bx, by, all_pos):
    if bx is None or by is None:
        return "None"
    for rod_num, positions in all_pos.items():
        rx = ROD_CONFIG[rod_num]['x']
        for idx, py in enumerate(positions):
            if check_ball_in_hit_zone(bx, by, rx, py):
                return get_player_id(rod_num, idx)
    return "None"


def find_peaks_above_threshold(inverted, y_positions, threshold):
    is_peak = inverted > threshold
    diff = np.diff(is_peak.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    regions = []
    for s, e in zip(starts, ends):
        s, e = max(0, s), min(len(y_positions) - 1, e)
        regions.append({
            'y_start': y_positions[s], 'y_end': y_positions[e],
            'y_center': (y_positions[s] + y_positions[e]) / 2,
            'width_mm': y_positions[e] - y_positions[s],
            'height': np.max(inverted[s:e + 1]) if e > s else inverted[s]
        })
    return regions


def merge_nearby_regions(regions, merge_gap=10, complete_width=45):
    if not regions:
        return []
    regions.sort(key=lambda r: r['y_start'])
    merged = [regions[0].copy()]
    for r in regions[1:]:
        if (merged[-1]['width_mm'] < complete_width and r['y_start'] - merged[-1]['y_end'] <= merge_gap):
            merged[-1]['y_end'] = r['y_end']
            merged[-1]['height'] = max(merged[-1]['height'], r['height'])
        else:
            merged[-1]['y_center'] = (merged[-1]['y_start'] + merged[-1]['y_end']) / 2
            merged[-1]['width_mm'] = merged[-1]['y_end'] - merged[-1]['y_start']
            merged.append(r.copy())
    merged[-1]['y_center'] = (merged[-1]['y_start'] + merged[-1]['y_end']) / 2
    merged[-1]['width_mm'] = merged[-1]['y_end'] - merged[-1]['y_start']
    return merged


def find_stoppers(regions, player_color):
    req_width = 35 if player_color == 'dark' else 10
    valid = sorted([r for r in regions if r['width_mm'] >= req_width], key=lambda p: p['height'], reverse=True)
    if len(valid) < 2:
        return None, None
    return tuple(sorted(valid[:2], key=lambda p: p['y_center']))


def detect_rod(intensities, y_positions, rod_num, config):
    inverted = 255 - intensities
    threshold = np.percentile(inverted, 50) + 30
    regions = find_peaks_above_threshold(inverted, y_positions, threshold)
    merged = merge_nearby_regions(regions)
    top, bottom = find_stoppers(merged, config['player_color'])
    if not top or not bottom:
        return []
    if config['offsets_bottom'] is None:
        return [(top['y_center'] + bottom['y_center']) / 2]
    if FIELD_HEIGHT_MM - bottom['y_end'] <= top['y_start']:
        poi, offsets = bottom['y_start'], config['offsets_bottom']
    else:
        poi, offsets = top['y_end'], config['offsets_top']
    return sorted([poi + o for o in offsets])


def init_zone_states(all_player_positions):
    states = {}
    for rod_num, positions in all_player_positions.items():
        for idx in range(len(positions)):
            pid = get_player_id(rod_num, idx)
            states[pid] = {
                'active': False, 'entry_vx': 0.0, 'entry_vy': 0.0,
                'entry_speed': 0.0, 'entry_frame': -1,
                'frames_inside': 0, 'min_speed_inside': 999.0, 'was_stopped': False
            }
    return states



# Main per-frame processing loop

def process_video(video_path, K_new, map1, map2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1 / 30.0
    out_w = int(FIELD_WIDTH_MM + 2 * PAD)
    out_h = int(FIELD_HEIGHT_MM + 2 * PAD)

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    H_matrix = rvec_global = tvec_global = None
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        img_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        H_matrix, rvec_global, tvec_global = compute_homography_and_pose(img_undist, K_new)
        if H_matrix is not None:
            break

    if H_matrix is None:
        cap.release()
        raise ValueError("Could not detect ArUco markers / compute homography in the first 100 frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracking_results = {}
    hit_frames = set()
    goal_frames_black = set()
    goal_frames_white = set()

    prev_x = prev_y = None
    prev_v = prev_vx = prev_vy = 0.0
    prev_status = "Searching"
    goal_counter = 0
    zone_states = {}
    last_onfield_x = last_onfield_y = None
    stable_positions = {r: None for r in range(1, 9)}

    def smart_stabilize_local(rod_num, new_pos):
        if len(new_pos) == 0:
            return stable_positions[rod_num] if stable_positions[rod_num] else []
        if stable_positions[rod_num] is None or len(stable_positions[rod_num]) != len(new_pos):
            stable_positions[rod_num] = list(new_pos)
            return new_pos
        if max(abs(new_pos[i] - stable_positions[rod_num][i]) for i in range(len(new_pos))) <= MIN_MOVEMENT_MM:
            return stable_positions[rod_num]
        stable_positions[rod_num] = list(new_pos)
        return new_pos

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num * dt
        frame_hit_player = "No"
        frame_hit_reason = "No"

        img_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(img_undist, H_matrix, (out_w, out_h))

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        measured_pt = None
        detected_x_mm = detected_y_mm = None
        for cnt in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if 5 < radius < 35:
                measured_pt = np.array([[np.float32(x)], [np.float32(y)]])
                detected_x_mm = int(x) - PAD
                detected_y_mm = int(y) - PAD
                break

        predicted = kf.predict()
        pred_x_raw = int(predicted[0][0])
        pred_y_raw = int(predicted[1][0])

        status = "Searching"
        log_x = log_y = None

        if measured_pt is not None:
            kf.correct(measured_pt)
            if 0 <= detected_x_mm <= FIELD_WIDTH_MM and 0 <= detected_y_mm <= FIELD_HEIGHT_MM:
                goal_counter = 0
                status, log_x, log_y = "On Field", detected_x_mm, detected_y_mm
                last_onfield_x, last_onfield_y = detected_x_mm, detected_y_mm
            else:
                near_left_goal = detected_x_mm < 0 and abs(detected_y_mm - GOAL_Y_CENTER) < GOAL_TOLERANCE
                near_right_goal = detected_x_mm > FIELD_WIDTH_MM and abs(detected_y_mm - GOAL_Y_CENTER) < GOAL_TOLERANCE
                if near_left_goal or near_right_goal:
                    goal_counter += 1
                    if goal_counter > 5:
                        status, log_x, log_y = "Goal", "Goal", "Goal"
                    else:
                        status, log_x, log_y = "Wall", "Wall", "Wall"
                else:
                    goal_counter = 0
                    status, log_x, log_y = "Wall", "Wall", "Wall"
        else:
            pred_x_mm = pred_x_raw - PAD
            pred_y_mm = pred_y_raw - PAD
            if (last_onfield_x is not None and
                    (last_onfield_x < 50 or last_onfield_x > FIELD_WIDTH_MM - 50) and
                    abs(last_onfield_y - GOAL_Y_CENTER) < GOAL_TOLERANCE):
                goal_counter += 1
                if goal_counter > 5:
                    status, log_x, log_y = "Goal", "Goal", "Goal"
            if status != "Goal":
                if 0 <= pred_x_mm <= FIELD_WIDTH_MM and 0 <= pred_y_mm <= FIELD_HEIGHT_MM:
                    status, log_x, log_y = "Occluded", pred_x_mm, pred_y_mm

        if status == "Goal" and prev_status != "Goal":
            if prev_x is not None and isinstance(prev_x, int):
                if prev_x < 50:
                    goal_frames_white.add(frame_num)
                elif prev_x > FIELD_WIDTH_MM - 50:
                    goal_frames_black.add(frame_num)
        prev_status = status

        ball_x_mm = ball_y_mm = None
        if isinstance(log_x, int):
            ball_x_mm, ball_y_mm = log_x, log_y

        curr_v = accel = curr_vx = curr_vy = 0.0
        if isinstance(log_x, int) and prev_x is not None and isinstance(prev_x, int):
            dx, dy = log_x - prev_x, log_y - prev_y
            curr_v = math.sqrt(dx ** 2 + dy ** 2) / 1000.0 / dt
            curr_vx = dx / 1000.0 / dt
            curr_vy = dy / 1000.0 / dt
            accel = (curr_v - prev_v) / dt

        if isinstance(log_x, int):
            prev_x, prev_y, prev_v = log_x, log_y, curr_v

        all_player_positions = {}
        for rod_num, config in ROD_CONFIG.items():
            points, y_pos = project_rod_line_to_warped(config['x'], rvec_global, tvec_global, K_new, H_matrix)
            intensities = extract_intensities_along_line(warped, points)
            smoothed = apply_bilateral_filter_1d(intensities)
            player_pos = smart_stabilize_local(rod_num, detect_rod(smoothed, y_pos, rod_num, config))
            all_player_positions[rod_num] = player_pos

        if not zone_states:
            zone_states = init_zone_states(all_player_positions)

        current_hit_zone = find_ball_in_hit_zones(ball_x_mm, ball_y_mm, all_player_positions)

        for rod_num, config in ROD_CONFIG.items():
            rx = config['x']
            positions = all_player_positions.get(rod_num, [])
            for pidx, py in enumerate(positions):
                pid = get_player_id(rod_num, pidx)
                if pid not in zone_states:
                    zone_states[pid] = {
                        'active': False, 'entry_vx': 0.0, 'entry_vy': 0.0,
                        'entry_speed': 0.0, 'entry_frame': -1,
                        'frames_inside': 0, 'min_speed_inside': 999.0, 'was_stopped': False
                    }
                in_zone_now = check_ball_in_hit_zone(ball_x_mm, ball_y_mm, rx, py)
                was_active = zone_states[pid]['active']

                if not was_active and in_zone_now:
                    zone_states[pid].update({
                        'active': True, 'entry_vx': curr_vx, 'entry_vy': curr_vy,
                        'entry_speed': curr_v, 'entry_frame': frame_num,
                        'frames_inside': 1, 'min_speed_inside': curr_v,
                        'was_stopped': curr_v < STOPPED_THRESHOLD
                    })
                elif was_active and in_zone_now:
                    zone_states[pid]['frames_inside'] += 1
                    if curr_v < zone_states[pid]['min_speed_inside']:
                        zone_states[pid]['min_speed_inside'] = curr_v
                    if curr_v < STOPPED_THRESHOLD:
                        zone_states[pid]['was_stopped'] = True
                elif was_active and not in_zone_now:
                    zone_states[pid]['active'] = False
                    entry_speed = zone_states[pid]['entry_speed']
                    entry_vx = zone_states[pid]['entry_vx']
                    entry_vy = zone_states[pid]['entry_vy']
                    exit_speed = curr_v
                    exit_vx, exit_vy = curr_vx, curr_vy
                    frames_inside = zone_states[pid]['frames_inside']
                    was_stopped = zone_states[pid]['was_stopped']

                    hit_type = None
                    direction_change = 0.0
                    if entry_speed > STOPPED_THRESHOLD and exit_speed > STOPPED_THRESHOLD:
                        cos_a = max(-1, min(1, (entry_vx * exit_vx + entry_vy * exit_vy) / (entry_speed * exit_speed)))
                        direction_change = math.degrees(math.acos(cos_a))

                    if frames_inside < MIN_FRAMES_CONTROL:
                        if direction_change > BLOCK_ANGLE:
                            hit_type = f"Block {direction_change:.0f}deg"
                        elif direction_change > MIN_DIRECTION_CHANGE:
                            hit_type = f"Brush {direction_change:.0f}deg"
                    else:
                        if was_stopped:
                            if exit_speed >= FAST_EXIT_THRESHOLD:
                                hit_type = "Strong Kick"
                            elif exit_speed >= SLOW_EXIT_THRESHOLD:
                                hit_type = "Weak Kick"
                            else:
                                hit_type = "Stop (No Kick)"
                        else:
                            if exit_speed >= FAST_EXIT_THRESHOLD:
                                hit_type = "Strong Dribble & Kick"
                            elif exit_speed >= SLOW_EXIT_THRESHOLD:
                                hit_type = "Weak Dribble & Kick"
                            else:
                                hit_type = "Dribble (No Kick)"

                    if hit_type and frame_num not in hit_frames:
                        hit_frames.add(frame_num)
                        frame_hit_player = pid
                        frame_hit_reason = hit_type

        prev_vx, prev_vy = curr_vx, curr_vy

        goals_black = len(goal_frames_black)
        goals_white = len(goal_frames_white)

        expected_players = {1: 1, 2: 2, 3: 3, 4: 5, 5: 5, 6: 3, 7: 2, 8: 1}
        rod_positions = {}
        for rod_num in range(1, 9):
            positions = all_player_positions.get(rod_num, [])
            for p_idx in range(expected_players[rod_num]):
                col = f'y_rod_{rod_num}_p{p_idx + 1}'
                rod_positions[col] = round(float(positions[p_idx]) / FIELD_HEIGHT_MM, 4) if p_idx < len(positions) else -1

        x_ball_norm = round(float(log_x) / FIELD_WIDTH_MM, 4) if isinstance(log_x, int) else 0.5
        y_ball_norm = round(float(log_y) / FIELD_HEIGHT_MM, 4) if isinstance(log_y, int) else 0.5
        ball_detected = 1 if status in ('On Field', 'Occluded') else 0
        frame_norm = round(frame_num / 1800, 4)

        tracking_results[frame_num] = {
            'Frame': frame_num, 'Timestamp': round(timestamp, 4),
            'X': log_x, 'Y': log_y,
            'Speed_m_s': round(curr_v, 3), 'Accel_m_s2': round(accel, 3),
            'Status': status,
            'Goals_Black': goals_black, 'Goals_White': goals_white,
            'Hit_Zone': current_hit_zone,
            'Hit_Successful': "Yes" if frame_hit_player != "No" else "No",
            'Hit_Player': frame_hit_player, 'Hit_Reason': frame_hit_reason,
            **rod_positions,
            'x_ball': x_ball_norm, 'y_ball': y_ball_norm,
            'ball_detected': ball_detected, 'frame_norm': frame_norm,
        }
        frame_num += 1

    cap.release()

    sorted_results = [tracking_results[k] for k in sorted(tracking_results.keys())]
    df = pd.DataFrame(sorted_results)

    goals_black = len(goal_frames_black)
    goals_white = len(goal_frames_white)

    return df, {
        "frames_processed": frame_num,
        "hits_detected": len(hit_frames),
        "goals_black": goals_black,
        "goals_white": goals_white,
    }



# Sequence building 

N_SEQUENCES = 6

FEATURE_COLS = [
    'x_ball', 'y_ball',
    'y_rod_1_p1',
    'y_rod_2_p1', 'y_rod_2_p2',
    'y_rod_3_p1', 'y_rod_3_p2', 'y_rod_3_p3',
    'y_rod_4_p1', 'y_rod_4_p2', 'y_rod_4_p3', 'y_rod_4_p4', 'y_rod_4_p5',
    'y_rod_5_p1', 'y_rod_5_p2', 'y_rod_5_p3', 'y_rod_5_p4', 'y_rod_5_p5',
    'y_rod_6_p1', 'y_rod_6_p2', 'y_rod_6_p3',
    'y_rod_7_p1', 'y_rod_7_p2',
    'y_rod_8_p1',
]


def build_sequences(df):
    df = df[df['Status'].isin(['On Field', 'Occluded'])].reset_index(drop=True)
    n_clean = len(df)
    if n_clean < N_SEQUENCES:
        raise ValueError(f"Only {n_clean} clean frames detected - not enough for {N_SEQUENCES} sequences. "
                          "Check that the ball is being tracked correctly.")

    seq_len = n_clean // N_SEQUENCES
    sequences = []
    for i in range(N_SEQUENCES):
        start, end = i * seq_len, i * seq_len + seq_len
        seq_df = df.iloc[start:end].copy().reset_index(drop=True)
        n = len(seq_df)
        seq_df['frame_norm'] = np.arange(n) / (n - 1) if n > 1 else np.zeros(n)
        feature_data = seq_df[FEATURE_COLS].values
        frame_norm = seq_df['frame_norm'].values.reshape(-1, 1)
        seq_array = np.hstack([feature_data, frame_norm])
        sequences.append(seq_array.astype(np.float32))
    return sequences



# TCN model 

N_FEATURES = 25
N_CLASSES = 3
LABEL_NAMES = {0: 'White', 1: 'Black', 2: 'Draw'}


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.shape[2]]
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = out[:, :, :x.shape[2]]
        out = self.bn2(out)
        out = F.relu(out)
        out = self.drop2(out)
        return F.relu(out + self.residual(x))


class TCN(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES, n_channels=16, kernel_size=3, n_blocks=3):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_blocks):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, dilation))
            in_ch = n_channels
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=2)
        return self.classifier(out)



# End-to-end prediction

def load_calibration(calib_dir):
    K_new = np.load(os.path.join(calib_dir, "newK.npy"))
    map1 = np.load(os.path.join(calib_dir, "map1.npy"))
    map2 = np.load(os.path.join(calib_dir, "map2.npy"))
    return K_new, map1, map2


def load_model(weights_path):
    model = TCN(n_features=N_FEATURES, n_classes=N_CLASSES)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_winner(video_path, K_new, map1, map2, model):
    df, stats = process_video(video_path, K_new, map1, map2)
    sequences = build_sequences(df)

    X = np.stack(sequences)  # (6, seq_len, 25)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (6, 25, seq_len) for Conv1d

    with torch.no_grad():
        logits = model(X_t)
        probs = F.softmax(logits, dim=1).numpy()  # (6, 3)
        preds = probs.argmax(axis=1)  # (6,)

    # Hard vote: majority of per-sequence predictions
    votes = np.bincount(preds, minlength=3)
    hard_pred = int(votes.argmax())

    # Soft vote: average predicted probabilities across sequences
    avg_probs = probs.mean(axis=0)
    soft_pred = int(avg_probs.argmax())

    return {
        "winner_hard_vote": LABEL_NAMES[hard_pred],
        "winner_soft_vote": LABEL_NAMES[soft_pred],
        "hard_vote_counts": {LABEL_NAMES[i]: int(votes[i]) for i in range(3)},
        "soft_vote_probabilities": {LABEL_NAMES[i]: round(float(avg_probs[i]), 4) for i in range(3)},
        "per_sequence_predictions": [LABEL_NAMES[int(p)] for p in preds],
        **stats,
    }
