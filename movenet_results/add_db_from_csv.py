import pandas as pd
from app.database import SessionLocal
from app.models import UserProfile, Feature

# CSV 로드 및 병합
df_ts = pd.read_csv("movenet_analysis_summary_ts.csv")
df_vs = pd.read_csv("movenet_analysis_summary_vs.csv")
df_all = pd.concat([df_ts, df_vs], ignore_index=True)

# 성별 한글 변환 + 연령대 파싱
gender_map = {"M": "남자", "F": "여자"}
df_all["gender"] = df_all["Group_ID"].str[0].map(gender_map)
df_all["age"] = df_all["Group_ID"].str[1].astype(int) * 10

# DB 세션 시작
session = SessionLocal()

for _, row in df_all.iterrows():
    # 각 row마다 새로운 사용자 생성
    profile = UserProfile(
        gender=row["gender"],  # "남자" 또는 "여자"
        age=row["age"],        # 10, 20, ...
        weight=None,
        height=None
    )
    session.add(profile)
    session.flush()  # profile.id 확보

    feature = Feature(
        user_profile_id=profile.id,
        image_filename=row["Image_Filename"],
        shoulder_height_diff_px=row["Shoulder_Height_Diff_px"],
        hip_height_diff_px=row["Hip_Height_Diff_px"],
        torso_vertical_tilt_deg=row["Torso_Vertical_Tilt_deg"],
        ear_hip_vertical_tilt_deg=row["Ear_Hip_Vertical_Tilt_deg"],
        shoulder_line_horizontal_tilt_deg=row["Shoulder_Line_Horizontal_Tilt_deg"],
        hip_line_horizontal_tilt_deg=row["Hip_Line_Horizontal_Tilt_deg"]
    )
    session.add(feature)

try:
    session.commit()
    print("모든 데이터를 성별 '남자/여자'로 변환하여 저장 완료!")
except Exception as e:
    session.rollback()
    print(f"저장 중 오류 발생: {e}")
finally:
    session.close()