from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models import Base, User, UserProfile
from app.utils import hash_password, verify_password
import shutil
import os

app = FastAPI()
Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 요청 모델
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

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
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
    hashed = hash_password(req.password)
    new_user = User(email=email, password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "회원가입 완료", "user_id": new_user.id}

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
def upload_image(user_id: int, file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, f"user_{user_id}_{file.filename}")
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "이미지 업로드 완료", "file_path": file_location}