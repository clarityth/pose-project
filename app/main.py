import datetime
from datetime import timezone, timedelta

import numpy as np
import openai
import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import shutil, os
from pathlib import Path

import tensorflow as tf

from app.database import SessionLocal, engine
from app.models import Base, User, UserProfile, Feature
from app.utils import hash_password, verify_password
from app.pose_utils import (
    process_image_with_movenet,
    extract_metrics_and_coords,
    save_keypoints_image,
    save_lines_image,
    COLOR_S_H_LINES,
    COLOR_TORSO_TILT,
    COLOR_EAR_HIP_TILT,
)
import joblib

from app.schemas import UploadResponse

app = FastAPI()
Base.metadata.create_all(bind=engine)

# 정적 파일(serving) 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../uploaded_images")
TMP_OUTPUT_DIR = Path("static/pose_results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent

# 포즈 추정 MoveNet TFLite 모델 로드
MODEL_DIR = BASE_DIR / "pose_model"
MOVENET_TFLITE_MODEL_PATH = MODEL_DIR / "movenet_thunder.tflite"
if not MOVENET_TFLITE_MODEL_PATH.exists():
    raise RuntimeError(
        f"MoveNet TFLite 모델을 찾을 수 없습니다: {MOVENET_TFLITE_MODEL_PATH}\n"
        "pose_model 디렉터리에 movenet_thunder.tflite 파일이 있는지 확인하세요."
    )
interpreter = tf.lite.Interpreter(model_path=str(MOVENET_TFLITE_MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ML RandomForest 모델(스케일러, 랜덤포레스트, 레이블 인코더) 로드
SCALER_PATH        = BASE_DIR / "model" / "exercise_scaler.pkl"
MODEL_PATH         = BASE_DIR / "model" / "exercise_rf_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "model" / "exercise_label_encoder.pkl"

scaler        = joblib.load(str(SCALER_PATH))
rf_model      = joblib.load(str(MODEL_PATH))
label_encoder = joblib.load(str(LABEL_ENCODER_PATH))

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 요청 모델 정의
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProfileRequest(BaseModel):
    email: str
    gender: str
    age: int
    weight: int
    height: int

@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email = str(req.email)
    name = req.name
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
    hashed = hash_password(req.password)
    new_user = User(email=email, password=hashed, name=name)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "회원가입 완료", "user_id": new_user.id, "name": new_user.name}


@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = str(req.email)
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")
    return {"message": "로그인 성공", "user_id": user.id, "name": user.name}


@app.post("/profile")
def create_profile(req: ProfileRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자 없음")
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile:
        # 이미 등록되어 있으면 값만 업데이트
        profile.gender = req.gender
        profile.age = req.age
        profile.weight = req.weight
        profile.height = req.height
        db.commit()
        return {"message": "신체 정보가 수정되었습니다."}
    else:
        # 등록되어 있지 않으면 새로 생성
        profile = UserProfile(
            user_id=user.id,
            gender=req.gender,
            age=req.age,
            weight=req.weight,
            height=req.height
        )
        db.add(profile)
        db.commit()
        return {"message": "신체 정보 등록 완료"}

def none_to_zero(val):
    return 0.0 if val is None else val

@app.post(
    "/upload-image",
    response_model=UploadResponse,
    summary="이미지 업로드 → 포즈 추정 + 지표 계산 + RandomForest 운동 추천 + LLM 리포트 생성",
    )

async def upload_image(
    email: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 이미지 파일 저장
    dest_path = UPLOAD_DIR / f"user_{email}_{file.filename}"
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 사용자 프로필 조회
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자 없음")
    profile = db.query(UserProfile).filter_by(user_id=user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="사용자 프로필이 없습니다.")

    # 포즈 추정
    keypoints, (w, h), orig_img = process_image_with_movenet(
        dest_path, interpreter, input_details, output_details
    )
    if keypoints is None or orig_img is None:
        raise HTTPException(status_code=400, detail="이미지 처리 실패")

    # 지표 계산
    analysis = extract_metrics_and_coords(keypoints, w, h)
    metrics = analysis["metrics"]
    coords = analysis["points_coords"]

    # 지표 각도 값을 절대값으로 변환
    for deg_key in [
        "torso_vertical_tilt",
        "ear_hip_vertical_tilt",
        "shoulder_line_horizontal_tilt",
        "hip_line_horizontal_tilt"
    ]:
        if metrics.get(deg_key) is not None:
            metrics[deg_key] = abs(metrics[deg_key])

    # 이전 업로드된 지표와의 변화량 계산
    prev_feature = (
        db.query(Feature)
        .filter(Feature.user_profile_id == profile.id)
        .order_by(Feature.id.desc())
        .offset(1)
        .first()
    )
    changes = {}
    if prev_feature:
        for key, db_field in [
            ("shoulder_height_diff_px", "shoulder_height_diff_px"),
            ("hip_height_diff_px", "hip_height_diff_px"),
            ("torso_vertical_tilt_deg", "torso_vertical_tilt_deg"),
            ("ear_hip_vertical_tilt_deg", "ear_hip_vertical_tilt_deg"),
            ("shoulder_line_horizontal_tilt_deg", "shoulder_line_horizontal_tilt_deg"),
            ("hip_line_horizontal_tilt_deg", "hip_line_horizontal_tilt_deg"),
        ]:
            old_val = getattr(prev_feature, db_field)
            if key.endswith("_px"):
                new_val = metrics[key.replace("_px", "")]
            else:
                new_val = metrics[key.replace("_deg", "")]
            changes[key] = (
                round(new_val - old_val, 2)
                if (new_val is not None and old_val is not None)
                else None
            )
    else:
        changes = {
            key: None
            for key in [
                "shoulder_height_diff_px",
                "hip_height_diff_px",
                "torso_vertical_tilt_deg",
                "ear_hip_vertical_tilt_deg",
                "shoulder_line_horizontal_tilt_deg",
                "hip_line_horizontal_tilt_deg",
            ]
        }

    # 포즈 추정 시각화 이미지 저장
    stem = dest_path.stem
    kp_img = TMP_OUTPUT_DIR / f"{stem}_keypoints.png"
    sh_img = TMP_OUTPUT_DIR / f"{stem}_shoulder_hip.png"
    torso_img = TMP_OUTPUT_DIR / f"{stem}_torso_tilt.png"
    ear_img = TMP_OUTPUT_DIR / f"{stem}_ear_hip_tilt.png"

    save_keypoints_image(orig_img, keypoints, w, h, kp_img)
    save_lines_image(
        orig_img,
        coords,
        [("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP")],
        COLOR_S_H_LINES,
        sh_img,
    )
    save_lines_image(
        orig_img,
        coords,
        [("shoulder_midpoint", "hip_midpoint")],
        COLOR_TORSO_TILT,
        torso_img,
    )
    save_lines_image(
        orig_img,
        coords,
        [("ear_midpoint", "hip_midpoint")],
        COLOR_EAR_HIP_TILT,
        ear_img,
    )

    # 6개 지표의 평균으로 종합점수 계산
    metric_values = [
        metrics["shoulder_height_diff"],
        metrics["hip_height_diff"],
        metrics["torso_vertical_tilt"],
        metrics["ear_hip_vertical_tilt"],
        metrics["shoulder_line_horizontal_tilt"],
        metrics["hip_line_horizontal_tilt"],
    ]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in metric_values):
        composite_score_current = None
    else:
        composite_score_current = sum(metric_values) / len(metric_values)

    # DB에 Feature 저장
    feature = Feature(
        user_profile_id=profile.id,
        image_filename=str(dest_path),
        shoulder_height_diff_px=metrics["shoulder_height_diff"],
        hip_height_diff_px=metrics["hip_height_diff"],
        torso_vertical_tilt_deg=metrics["torso_vertical_tilt"],
        ear_hip_vertical_tilt_deg=metrics["ear_hip_vertical_tilt"],
        shoulder_line_horizontal_tilt_deg=metrics["shoulder_line_horizontal_tilt"],
        hip_line_horizontal_tilt_deg=metrics["hip_line_horizontal_tilt"],
        composite_score=composite_score_current,
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)

    # 동일 그룹(성별, 나이) 전체의 종합점수 평균·표준편차·퍼센타일·최솟값·최댓값 계산
    comp_rows = (
        db.query(Feature.composite_score)
          .join(UserProfile)
          .filter(
              UserProfile.gender == profile.gender,
              UserProfile.age == profile.age,
              Feature.composite_score != None,
          )
          .all()
    )
    comp_list = [row[0] for row in comp_rows if row[0] is not None]

    if len(comp_list) > 0:
        # 평균
        composite_mean = float(np.mean(comp_list))
        # 표준편차
        composite_std  = float(np.std(comp_list, ddof=0))

        # 퍼센타일
        if composite_score_current is not None:
            rank = np.sum(np.array(comp_list) <= composite_score_current) / len(comp_list) * 100
            composite_percentile = round(float(rank), 1)
        else:
            composite_percentile = None

        # min, max
        composite_min = float(np.min(comp_list))
        composite_max = float(np.max(comp_list))
    else:
        composite_mean       = None
        composite_std        = None
        composite_percentile = None
        composite_min        = None
        composite_max        = None

    # 이번 업로드 포함 최근 5개 업로드 조회
    recent_objs = (
        db.query(Feature)
        .filter(Feature.user_profile_id == profile.id)
        .order_by(Feature.created_at.desc())
        .limit(5)
        .all()
    )
    # 업로드 날짜 오름차순 정렬
    recent_five = sorted(recent_objs, key=lambda f: f.created_at)

    history_list = []
    base_url = os.getenv("BASE_URL")
    for feat in recent_five:
        # 이미지 URL 만들기
        stem_hist = Path(feat.image_filename).stem
        visuals = {
            "keypoints": f"{base_url}/static/pose_results/{stem_hist}_keypoints.png",
            "shoulder_hip": f"{base_url}/static/pose_results/{stem_hist}_shoulder_hip.png",
            "torso_tilt": f"{base_url}/static/pose_results/{stem_hist}_torso_tilt.png",
            "ear_hip_tilt": f"{base_url}/static/pose_results/{stem_hist}_ear_hip_tilt.png",
        }
        vals = [
            feat.shoulder_height_diff_px,
            feat.hip_height_diff_px,
            feat.torso_vertical_tilt_deg,
            feat.ear_hip_vertical_tilt_deg,
            feat.shoulder_line_horizontal_tilt_deg,
            feat.hip_line_horizontal_tilt_deg,
        ]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            composite_score_history = None
        else:
            composite_score_history = sum(vals) / len(vals)

        created_at = feat.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone(timedelta(hours=9)))
        history_list.append({
            "feature_id": feat.id,
            "shoulder_height_diff_px": none_to_zero(feat.shoulder_height_diff_px),
            "hip_height_diff_px": none_to_zero(feat.hip_height_diff_px),
            "torso_vertical_tilt_deg": none_to_zero(feat.torso_vertical_tilt_deg),
            "ear_hip_vertical_tilt_deg": none_to_zero(feat.ear_hip_vertical_tilt_deg),
            "shoulder_line_horizontal_tilt_deg": none_to_zero(feat.shoulder_line_horizontal_tilt_deg),
            "hip_line_horizontal_tilt_deg": none_to_zero(feat.hip_line_horizontal_tilt_deg),
            "composite_score": composite_score_history,
            "created_at": created_at.isoformat(),
            "visuals": visuals,
        })

    # 마지막 업로드 날짜, 경과 일수 계산
    last_feature = prev_feature or feature
    created_at_kst = last_feature.created_at
    if created_at_kst.tzinfo is None:
        created_at_kst = created_at_kst.replace(tzinfo=timezone(timedelta(hours=9)))
    last_date_iso = created_at_kst.isoformat()
    now_kst = datetime.datetime.now(timezone(timedelta(hours=9)))
    days_since = (now_kst - created_at_kst).days

    # RandomForest 모델 운동 추천
    gender_map = {"남자": 0, "여자": 1}
    age_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}
    g_code = gender_map.get(profile.gender, 0)
    a_code = age_map.get(profile.age, 0)

    feature_vector = [
        g_code,
        a_code,
        metrics["shoulder_height_diff"],
        metrics["hip_height_diff"],
        metrics["torso_vertical_tilt"],
        metrics["ear_hip_vertical_tilt"],
        metrics["shoulder_line_horizontal_tilt"],
        metrics["hip_line_horizontal_tilt"],
    ]
    feature_vector = [
        0 if v is None or (isinstance(v, float) and np.isnan(v)) else v
        for v in feature_vector
    ]

    X_scaled = scaler.transform([feature_vector])
    proba = rf_model.predict_proba(X_scaled)[0]
    idxs = proba.argsort()[::-1][:3]
    recommendations = []
    for i in idxs:
        label = label_encoder.inverse_transform([i])[0]
        recommendations.append(
            {"exercise": label, "probability": round(float(proba[i]) * 100, 1)}
        )

    # YouTube 영상 검색
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY가 설정되어 있지 않습니다.")
    video_results = []
    for rec in recommendations:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": rec["exercise"],
                "type": "video",
                "maxResults": 3,
                "key": youtube_key,
            },
        )
        items = resp.json().get("items", [])
        if not items:
            continue
        snippet = items[0]["snippet"]
        vid_id = items[0]["id"]["videoId"]
        video_results.append(
            {
                "exercise": rec["exercise"],
                "video_title": snippet.get("title", ""),
                "video_desc": snippet.get("description", ""),
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "thumbnail_url": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            }
        )

    # LLM 종합 리포트 생성
    prompt = f"""
       사용자가 업로드한 이미지로부터 다음 지표가 계산되었습니다:
       - 어깨 높이 차이: {metrics['shoulder_height_diff']} px
       - 어깨 기울기: {metrics['shoulder_line_horizontal_tilt']}°
       - 골반 높이 차이: {metrics['hip_height_diff']} px
       - 골반 기울기: {metrics['hip_line_horizontal_tilt']}°
       - 몸통 수직 기울기: {metrics['torso_vertical_tilt']}°
       - 귀-골반 기울기: {metrics['ear_hip_vertical_tilt']}°
    
       ※ 픽셀(px)로 측정된 값은 반드시 “px” 단위로, 각도로 측정된 값은 반드시 “°” 단위로만 표기하세요.
          절대로 mm나 cm 등 다른 단위로 변환하지 마시고, 주어진 단위 그대로 쓰시기 바랍니다.
    
        - 종합점수 상위 퍼센타일: {composite_percentile}%
        (동일 성별·연령 그룹 내에서 상위 {composite_percentile}% 위치에 해당합니다. 퍼센타일이 낮을수록 체형이 더 좋은 상태입니다.)
    
       추천 운동과 확률:
       {recommendations}
    
       YouTube 영상 링크:
       {video_results}
    
       1. 위 “px”와 “°” 단위를 절대 존중하여, 다른 단위를 절대 사용하지 말 것.
       2. 단순히 수치를 나열하는 것을 넘어서, 전문가 관점의 의견(인사이트, 조언, 팁 등)을 적극적으로 추가할 것.
       3. “사용자”가 쉽게 이해할 수 있도록, 전문 용어는 최소화하고 부드러운 어투로 설명할 것.
    
       위 지침을 바탕으로, 사용자가 이해하기 쉬운 한글 “종합 체형 분석”과 “추천 운동” 리포트를 작성해주세요.
       """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY가 설정되어 있지 않습니다.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "당신은 온누리마취통증의학과의 자세 분석 AI 척추의 요정입니다."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    report = data["choices"][0]["message"]["content"]

    # Static URL 방식 이미지 반환
    visuals_urls = {
        "keypoints": f"{base_url}/static/pose_results/{stem}_keypoints.png",
        "shoulder_hip": f"{base_url}/static/pose_results/{stem}_shoulder_hip.png",
        "torso_tilt": f"{base_url}/static/pose_results/{stem}_torso_tilt.png",
        "ear_hip_tilt": f"{base_url}/static/pose_results/{stem}_ear_hip_tilt.png",
    }

    return {
        "message": "업로드, 분석, 운동추천, 영상추천, 리포트 생성 완료",
        "user": {
            "user_id": profile.user_id,
            "name": user.name if user else None,
            "gender": profile.gender,
            "age": profile.age,
            "weight": profile.weight,
            "height": profile.height,
        },
        "visuals": visuals_urls,
        "metrics": {
            "shoulder_height_diff_px": none_to_zero(metrics["shoulder_height_diff"]),
            "hip_height_diff_px": none_to_zero(metrics["hip_height_diff"]),
            "torso_vertical_tilt_deg": none_to_zero(metrics["torso_vertical_tilt"]),
            "ear_hip_vertical_tilt_deg": none_to_zero(metrics["ear_hip_vertical_tilt"]),
            "shoulder_line_horizontal_tilt_deg": none_to_zero(metrics["shoulder_line_horizontal_tilt"]),
            "hip_line_horizontal_tilt_deg": none_to_zero(metrics["hip_line_horizontal_tilt"]),
        },
        "composite_score_current": composite_score_current,
        "composite_percentile": composite_percentile,
        "composite_mean": composite_mean,
        "composite_std": composite_std,
        "composite_min": composite_min,
        "composite_max": composite_max,
        "last_upload_date": last_date_iso,
        "days_since": days_since,
        "changes": {k: none_to_zero(v) for k, v in changes.items()},
        "history": history_list,
        "recommendations": recommendations,
        "videos": video_results,
        "report": report,
    }