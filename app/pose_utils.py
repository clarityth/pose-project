import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import math

TMP_OUTPUT_DIR = Path("tmp")  # output dir

CONFIDENCE_THRESHOLD_FOR_CALC = 0.3 # 포즈 추정 threshold

COCO_KEYPOINT_NAMES = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

# line color
COLOR_KEYPOINT = (255, 0, 0)      # 파란색
COLOR_S_H_LINES = (0, 255, 0)     # 초록색 (어깨, 골반 선)
COLOR_TORSO_TILT = (0, 0, 255)    # 빨간색
COLOR_EAR_HIP_TILT = (255, 255, 0) # 청록색
LINE_THICKNESS = 2
POINT_RADIUS = 3


# calculate func 모델의 추출결과인 정규화된 좌표를 이미지 실제 크기에 맞춰 픽셀좌표로 변환
def get_keypoint_pixel_coords(normalized_keypoints_list, target_name, orig_w, orig_h, conf_thresh):
    for kp in normalized_keypoints_list:
        if kp.get("name") == target_name:
            score = kp.get("score", 0.0)
            if score >= conf_thresh:
                x_norm = kp.get("x_norm", -1.0)
                y_norm = kp.get("y_norm", -1.0)
                if x_norm == -1.0 or y_norm == -1.0:
                    return None
                return (int(x_norm * orig_w), int(y_norm * orig_h), score) # (x, y, score) tuple
    return None

def calculate_midpoint(p1_tuple, p2_tuple):
    # 두 점 (x,y,score 튜플)의 중점을 (x,y) 튜플로 반환. 하나라도 None이면 None 반환.
    if p1_tuple is None or p2_tuple is None:
        return None
    # score 제외하고 x, y 좌표만 사용
    return (int((p1_tuple[0] + p2_tuple[0]) / 2), int((p1_tuple[1] + p2_tuple[1]) / 2))

def calculate_height_difference_shoulders(l_shoulder_tuple, r_shoulder_tuple):
    # 좌우 어깨 높이 차이 계산
    if l_shoulder_tuple is None or r_shoulder_tuple is None:
        return None
    return abs(l_shoulder_tuple[1] - r_shoulder_tuple[1])

def calculate_height_difference_hips(l_hip_tuple, r_hip_tuple):
    # 좌우 골반 높이 차이 계산
    if l_hip_tuple is None or r_hip_tuple is None:
        return None
    return abs(l_hip_tuple[1] - r_hip_tuple[1])

def calculate_vertical_tilt_line(p1_tuple, p2_tuple):
    # 두 점을 잇는 선의 수직 기울기 (Y축 기준, 0도는 수직). p1이 위쪽 점으로 간주.
    if p1_tuple is None or p2_tuple is None:
        return None
    if p1_tuple[0] == p2_tuple[0] and p1_tuple[1] == p2_tuple[1]: # 두 점이 동일
        return 0.0

    delta_x = float(p2_tuple[0] - p1_tuple[0])
    delta_y = float(p2_tuple[1] - p1_tuple[1]) # 이미지 좌표계 (y가 아래로 증가)

    if delta_y == 0: # 수평선 (분모 0 방지)
        return 90.0 if delta_x != 0 else 0.0

    angle_rad = math.atan2(delta_x, delta_y)
    return math.degrees(angle_rad)

def calculate_horizontal_tilt_line(p1_tuple, p2_tuple):
    # 두 점을 잇는 선의 수평 기울기
    if p1_tuple is None or p2_tuple is None:
        return None
    if p1_tuple[0] == p2_tuple[0] and p1_tuple[1] == p2_tuple[1]: # 두 점이 동일
        return 0.0

    p1_final = p1_tuple
    p2_final = p2_tuple
    if p1_tuple[0] > p2_tuple[0]:
        p1_final = p2_tuple
        p2_final = p1_tuple

    delta_x_final = float(p2_final[0] - p1_final[0])
    delta_y_final = float(p2_final[1] - p1_final[1])

    if delta_x_final == 0:
        if delta_y_final > 0: return -90.0
        elif delta_y_final < 0: return 90.0
        else: return 0.0

    angle_rad_final = math.atan2(-delta_y_final, delta_x_final)
    angle_deg_final = math.degrees(angle_rad_final)

    return angle_deg_final


def extract_metrics_and_coords(normalized_keypoints_list, orig_w, orig_h):
    # 지표계산 관절 좌표
    metrics = {
        "shoulder_height_diff": None, "hip_height_diff": None,
        "torso_vertical_tilt": None, "ear_hip_vertical_tilt": None,
        "shoulder_line_horizontal_tilt": None, "hip_line_horizontal_tilt": None,
    }
    points_coords = {name: None for name in COCO_KEYPOINT_NAMES}
    points_coords.update({
        "shoulder_midpoint": None, "hip_midpoint": None, "ear_midpoint": None
    })

    l_shoulder_full = get_keypoint_pixel_coords(normalized_keypoints_list, "LEFT_SHOULDER", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)
    r_shoulder_full = get_keypoint_pixel_coords(normalized_keypoints_list, "RIGHT_SHOULDER", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)
    l_hip_full = get_keypoint_pixel_coords(normalized_keypoints_list, "LEFT_HIP", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)
    r_hip_full = get_keypoint_pixel_coords(normalized_keypoints_list, "RIGHT_HIP", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)
    l_ear_full = get_keypoint_pixel_coords(normalized_keypoints_list, "LEFT_EAR", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)
    r_ear_full = get_keypoint_pixel_coords(normalized_keypoints_list, "RIGHT_EAR", orig_w, orig_h, CONFIDENCE_THRESHOLD_FOR_CALC)

    if l_shoulder_full: points_coords["LEFT_SHOULDER"] = (l_shoulder_full[0], l_shoulder_full[1])
    if r_shoulder_full: points_coords["RIGHT_SHOULDER"] = (r_shoulder_full[0], r_shoulder_full[1])
    if l_hip_full: points_coords["LEFT_HIP"] = (l_hip_full[0], l_hip_full[1])
    if r_hip_full: points_coords["RIGHT_HIP"] = (r_hip_full[0], r_hip_full[1])
    if l_ear_full: points_coords["LEFT_EAR"] = (l_ear_full[0], l_ear_full[1])
    if r_ear_full: points_coords["RIGHT_EAR"] = (r_ear_full[0], r_ear_full[1])

    shoulder_midpoint = calculate_midpoint(l_shoulder_full, r_shoulder_full)
    hip_midpoint = calculate_midpoint(l_hip_full, r_hip_full)
    ear_midpoint = calculate_midpoint(l_ear_full, r_ear_full)

    if shoulder_midpoint: points_coords["shoulder_midpoint"] = shoulder_midpoint
    if hip_midpoint: points_coords["hip_midpoint"] = hip_midpoint
    if ear_midpoint: points_coords["ear_midpoint"] = ear_midpoint

    metrics["shoulder_height_diff"] = calculate_height_difference_shoulders(l_shoulder_full, r_shoulder_full)
    metrics["hip_height_diff"] = calculate_height_difference_hips(l_hip_full, r_hip_full)

    if shoulder_midpoint and hip_midpoint:
        metrics["torso_vertical_tilt"] = abs(calculate_vertical_tilt_line(shoulder_midpoint, hip_midpoint))
    if ear_midpoint and hip_midpoint:
        metrics["ear_hip_vertical_tilt"] = abs(calculate_vertical_tilt_line(ear_midpoint, hip_midpoint))

    metrics["shoulder_line_horizontal_tilt"] = abs(calculate_horizontal_tilt_line(l_shoulder_full, r_shoulder_full))
    metrics["hip_line_horizontal_tilt"] = abs(calculate_horizontal_tilt_line(l_hip_full, r_hip_full))

    return {"metrics": metrics, "points_coords": points_coords}



def process_image_with_movenet(image_path: Path, interpreter, input_details, output_details):
    # 이미지를 처리하고 정규화된 키포인트 리스트, 원본 크기, 원본 cv2 이미지를 반환.
    keypoints_normalized_list = []
    original_width, original_height = 0, 0
    original_cv2_image = None
    try:
        img_array = np.fromfile(str(image_path), np.uint8) # 한글 경로 처리
        original_cv2_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if original_cv2_image is None:
            print(f"    [MoveNet] 이미지 로드 실패: {image_path.name}")
            return None, (0, 0), None

        original_height, original_width = original_cv2_image.shape[:2]

        input_height = input_details[0]['shape'][1]
        input_width = input_details[0]['shape'][2]
        image_rgb = cv2.cvtColor(original_cv2_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_rgb, (input_width, input_height))

        input_data_type = input_details[0]['dtype']
        if input_data_type == np.float32:
            input_data = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)
        elif input_data_type == np.uint8:
            input_data = np.expand_dims(resized_image.astype(np.uint8), axis=0)
        else:
            print(f"    경고: MoveNet 입력 데이터 타입 {input_data_type} 미지원. uint8로 처리 시도.")
            input_data = np.expand_dims(resized_image.astype(np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        keypoints_output_tensor = interpreter.get_tensor(output_details[0]['index'])

        for i in range(keypoints_output_tensor.shape[2]):
            y_norm = float(keypoints_output_tensor[0, 0, i, 0])
            x_norm = float(keypoints_output_tensor[0, 0, i, 1])
            score = float(keypoints_output_tensor[0, 0, i, 2])
            keypoints_normalized_list.append({
                "index": i, "name": COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else f"UNKNOWN_{i}",
                "y_norm": y_norm, "x_norm": x_norm, "score": score
            })
        return keypoints_normalized_list, (original_width, original_height), original_cv2_image
    except Exception as e:
        print(f"    [MoveNet] 처리 중 오류 ({image_path.name}): {e}")
        return None, (original_width, original_height), original_cv2_image


def save_keypoints_image(original_image_cv2, normalized_keypoints_list, orig_w, orig_h, output_path: Path):
    # 관절 이미지 생성  
    if original_image_cv2 is None or not normalized_keypoints_list:
        print(f"Skipping keypoint image saving: No original image or keypoints ({output_path.name})")
        return

    image_to_draw = original_image_cv2.copy()
    for kp in normalized_keypoints_list:
        score = kp.get("score", 0.0)
        if score >= CONFIDENCE_THRESHOLD_FOR_CALC:
            x_norm = kp.get("x_norm", -1.0)
            y_norm = kp.get("y_norm", -1.0)
            if x_norm != -1.0 and y_norm != -1.0:
                px = int(x_norm * orig_w)
                py = int(y_norm * orig_h)
                cv2.circle(image_to_draw, (px, py), POINT_RADIUS, COLOR_KEYPOINT, -1)

    cv2.imwrite(str(output_path), image_to_draw)
    print(f"Keypoint image saved: {output_path}")

def save_lines_image(original_image_cv2, points_coords_dict, line_definitions, line_color, output_path: Path):
    #지표 이미지 생성
    if original_image_cv2 is None:
        print(f"Skipping line image saving: No original image ({output_path.name})")
        return

    image_to_draw = original_image_cv2.copy()
    valid_line_drawn = False
    for p1_name, p2_name in line_definitions:
        p1 = points_coords_dict.get(p1_name)
        p2 = points_coords_dict.get(p2_name)

        if p1 and p2:
            cv2.line(image_to_draw, p1, p2, line_color, LINE_THICKNESS)
            cv2.circle(image_to_draw, p1, POINT_RADIUS, line_color, -1)
            cv2.circle(image_to_draw, p2, POINT_RADIUS, line_color, -1)
            valid_line_drawn = True
        else:
            print(f"  Skipping line drawing for {output_path.name}: Point '{p1_name}' or '{p2_name}' not found.")

    if valid_line_drawn:
        cv2.imwrite(str(output_path), image_to_draw)
        print(f"Line image saved: {output_path}")
    else:
        print(f"Skipping line image saving: No valid lines to draw for {output_path.name}")