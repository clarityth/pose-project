import math
import datetime
from datetime import timezone, timedelta

import numpy as np
import openai
import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
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

app = FastAPI()
Base.metadata.create_all(bind=engine)

# StaticFiles 설정: "static" 디렉터리를 "/static" 경로로 제공
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

# MoveNet TFLite 모델 로드
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

# ML 모델(스케일러, 랜덤포레스트, 레이블 인코더) 로드
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
    user_id: int
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
    return {"message": "로그인 성공", "user_id": user.id}

@app.post("/profile")
def create_profile(req: ProfileRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자 없음")
    if db.query(UserProfile).filter(UserProfile.user_id == req.user_id).first():
        raise HTTPException(status_code=400, detail="이미 등록됨")
    profile = UserProfile(
        user_id=req.user_id,
        gender=req.gender,
        age=req.age,
        weight=req.weight,
        height=req.height
    )
    db.add(profile)
    db.commit()
    return {"message": "신체 정보 등록 완료"}

@app.post("/upload-image")
async def upload_image(
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1) 이미지 파일 저장
    dest_path = UPLOAD_DIR / f"user_{user_id}_{file.filename}"
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 2) 사용자 프로필 조회
    profile = db.query(UserProfile).filter_by(user_id=user_id).first()
    if not profile:
        raise HTTPException(404, "사용자 프로필이 없습니다.")
    # 2.1) 사용자 정보 조회
    user = db.query(User).filter(User.id == user_id).first()

    # 3) 포즈 추정
    keypoints, (w, h), orig_img = process_image_with_movenet(
        dest_path, interpreter, input_details, output_details
    )
    if keypoints is None or orig_img is None:
        raise HTTPException(400, "이미지 처리 실패")

    # 4) 지표 계산
    analysis = extract_metrics_and_coords(keypoints, w, h)
    metrics = analysis["metrics"]
    coords  = analysis["points_coords"]

    # 4.5) 이전 업로드된 지표 불러와 변화량 계산(2번째 최신 데이터)
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
            changes[key] = round(new_val - old_val, 2) if (new_val is not None and old_val is not None) else None
    else:
        changes = {key: None for key in [
            "shoulder_height_diff_px", "hip_height_diff_px",
            "torso_vertical_tilt_deg", "ear_hip_vertical_tilt_deg",
            "shoulder_line_horizontal_tilt_deg", "hip_line_horizontal_tilt_deg"
        ]}

    # 5) 동일 연령·성별 데이터 로드 및 백분위 계산
    df = (
        db.query(
            Feature.shoulder_height_diff_px,
            Feature.hip_height_diff_px,
            Feature.torso_vertical_tilt_deg,
            Feature.ear_hip_vertical_tilt_deg,
            Feature.shoulder_line_horizontal_tilt_deg,
            Feature.hip_line_horizontal_tilt_deg,
        )
        .join(UserProfile)
        .filter(
            UserProfile.gender == profile.gender,
            UserProfile.age    == profile.age
        )
        .all()
    )
    arr = np.array(df, dtype=float)  # shape (N,6)
    percentiles = {}
    for idx, key in enumerate([
        "shoulder_height_diff",
        "hip_height_diff",
        "torso_vertical_tilt",
        "ear_hip_vertical_tilt",
        "shoulder_line_horizontal_tilt",
        "hip_line_horizontal_tilt"
    ]):
        values = arr[:, idx]
        cur = metrics[key]
        if cur is None or math.isnan(cur):
            percentiles[key] = None
        else:
            rank = np.sum(values <= cur) / len(values) * 100
            percentiles[key] = round(rank, 1)

    # 6) 시각화 이미지 저장(StaticFiles 폴더 아래)
    stem = dest_path.stem
    kp_img    = TMP_OUTPUT_DIR / f"{stem}_keypoints.png"
    sh_img    = TMP_OUTPUT_DIR / f"{stem}_shoulder_hip.png"
    torso_img = TMP_OUTPUT_DIR / f"{stem}_torso_tilt.png"
    ear_img   = TMP_OUTPUT_DIR / f"{stem}_ear_hip_tilt.png"

    save_keypoints_image(orig_img, keypoints, w, h, kp_img)
    save_lines_image(orig_img, coords,
                     [("LEFT_SHOULDER","RIGHT_SHOULDER"),("LEFT_HIP","RIGHT_HIP")],
                     COLOR_S_H_LINES, sh_img)
    save_lines_image(orig_img, coords,
                     [("shoulder_midpoint","hip_midpoint")],
                     COLOR_TORSO_TILT, torso_img)
    save_lines_image(orig_img, coords,
                     [("ear_midpoint","hip_midpoint")],
                     COLOR_EAR_HIP_TILT, ear_img)

    # 7) DB에 Feature 저장
    feature = Feature(
        user_profile_id=profile.id,
        image_filename=str(dest_path),
        shoulder_height_diff_px=metrics["shoulder_height_diff"],
        hip_height_diff_px=metrics["hip_height_diff"],
        torso_vertical_tilt_deg=metrics["torso_vertical_tilt"],
        ear_hip_vertical_tilt_deg=metrics["ear_hip_vertical_tilt"],
        shoulder_line_horizontal_tilt_deg=metrics["shoulder_line_horizontal_tilt"],
        hip_line_horizontal_tilt_deg=metrics["hip_line_horizontal_tilt"],
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)

    # 8) 이번 업로드 포함 이전 기록 최대 5개만 조회
    all_features = (
        db.query(Feature)
        .filter(Feature.user_profile_id == profile.id)
        .order_by(Feature.created_at.asc())
        .limit(5)
        .all()
    )
    history_list = []
    for feat in all_features:
        created_at = feat.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone(timedelta(hours=9)))
        history_list.append({
            "feature_id": feat.id,
            "shoulder_height_diff_px": feat.shoulder_height_diff_px,
            "hip_height_diff_px": feat.hip_height_diff_px,
            "torso_vertical_tilt_deg": feat.torso_vertical_tilt_deg,
            "ear_hip_vertical_tilt_deg": feat.ear_hip_vertical_tilt_deg,
            "shoulder_line_horizontal_tilt_deg": feat.shoulder_line_horizontal_tilt_deg,
            "hip_line_horizontal_tilt_deg": feat.hip_line_horizontal_tilt_deg,
            "created_at": created_at.isoformat()
        })

    last_feature = prev_feature or feature
    created_at_kst = last_feature.created_at
    if created_at_kst.tzinfo is None:
        created_at_kst = created_at_kst.replace(tzinfo=timezone(timedelta(hours=9)))
    last_date_iso = created_at_kst.isoformat()
    now_kst = datetime.datetime.now(timezone(timedelta(hours=9)))
    days_since = (now_kst - created_at_kst).days

    # 9) RandomForest 모델 운동 추천
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
        recommendations.append({
            "exercise": label,
            "probability": round(float(proba[i]) * 100, 1)
        })

    # 10) YouTube 영상 검색
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_key:
        raise HTTPException(500, "YOUTUBE_API_KEY가 설정되어 있지 않습니다.")
    video_results = []
    for rec in recommendations:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": rec["exercise"],
                "type": "video",
                "maxResults": 3,
                "key": youtube_key
            }
        )
        items = resp.json().get("items", [])
        if not items:
            continue
        snippet = items[0]["snippet"]
        vid_id = items[0]["id"]["videoId"]
        video_results.append({
            "exercise": rec["exercise"],
            "video_title": snippet.get("title", ""),
            "video_desc": snippet.get("description", ""),
            "video_url": f"https://www.youtube.com/watch?v={vid_id}",
            "thumbnail_url": snippet.get("thumbnails", {})
            .get("medium", {})
            .get("url", "")
        })

    # 11) LLM 종합 리포트 생성
    prompt = f"""
    사용자가 업로드한 이미지로부터 다음 지표가 계산되었습니다:
    {metrics}

    백분위:
    {percentiles}
    백분위가 높을수록 해당 지표의 불균형이 크다는 의미이므로, 백분위가 높으면 개선이 필요함을 강조해주세요.

    추천 운동과 확률:
    {recommendations}

    YouTube 영상 링크:
    {video_results}

    위 정보를 바탕으로, 사용자에게 한글로 이해하기 쉬운 종합 체형 분석, 추천 운동 리포트를 마크다운이 아닌 단순 텍스트로 작성해주세요.
    """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(500, "OPENROUTER_API_KEY가 설정되어 있지 않습니다.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "당신은 온누리마취통증의학과의 자세 분석 AI 척추의 요정입니다."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    report = data["choices"][0]["message"]["content"]

    # 12) Static URL 방식으로 이미지 경로 반환
    base_url = os.getenv("BASE_URL")
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
            "shoulder_height_diff_px": metrics["shoulder_height_diff"],
            "hip_height_diff_px": metrics["hip_height_diff"],
            "torso_vertical_tilt_deg": metrics["torso_vertical_tilt"],
            "ear_hip_vertical_tilt_deg": metrics["ear_hip_vertical_tilt"],
            "shoulder_line_horizontal_tilt_deg": metrics["shoulder_line_horizontal_tilt"],
            "hip_line_horizontal_tilt_deg": metrics["hip_line_horizontal_tilt"],
        },
        "percentiles": {
            "shoulder_height_diff_px": percentiles["shoulder_height_diff"],
            "hip_height_diff_px": percentiles["hip_height_diff"],
            "torso_vertical_tilt": percentiles["torso_vertical_tilt"],
            "ear_hip_vertical_tilt": percentiles["ear_hip_vertical_tilt"],
            "shoulder_line_horizontal_tilt": percentiles["shoulder_line_horizontal_tilt"],
            "hip_line_horizontal_tilt": percentiles["hip_line_horizontal_tilt"],
        },
        "last_upload_date": last_date_iso,
        "days_since": days_since,
        "changes": changes,
        "history": history_list,
        "recommendations": recommendations,
        "videos": video_results,
        "report": report,
    }