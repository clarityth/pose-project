from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User, UserProfile
from utils import hash_password, verify_password

app = FastAPI()
Base.metadata.create_all(bind=engine)

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
    email: EmailStr
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
    email = str(req.email)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 이메일의 사용자가 없습니다.")
    if db.query(UserProfile).filter(UserProfile.user_id == user.id).first():
        raise HTTPException(status_code=400, detail="이미 프로필이 등록되어 있습니다.")
    profile = UserProfile(
        user_id=user.id,
        gender=req.gender,
        age=req.age,
        weight=req.weight,
        height=req.height
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return {"message": "신체 정보 등록 완료"}