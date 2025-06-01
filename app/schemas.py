from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    user_id: int = Field(..., description="사용자 고유 ID")
    name: Optional[str] = Field(None, description="사용자 이름")
    gender: str = Field(..., description="성별 (예: 남자, 여자)")
    age: int = Field(..., description="나이 (10,20,30 등)")
    weight: int = Field(..., description="몸무게(kg)")
    height: int = Field(..., description="키(cm)")


class Visuals(BaseModel):
    keypoints: str = Field(..., description="키포인트 이미지 URL")
    shoulder_hip: str = Field(..., description="어깨-골반 선 이미지 URL")
    torso_tilt: str = Field(..., description="몸통 기울기 이미지 URL")
    ear_hip_tilt: str = Field(..., description="귀-골반 기울기 이미지 URL")


class Metrics(BaseModel):
    shoulder_height_diff_px: float = Field(..., description="어깨 높이 차이 (px)")
    hip_height_diff_px: float = Field(..., description="골반 높이 차이 (px)")
    torso_vertical_tilt_deg: float = Field(..., description="몸통 수직 기울기 (°)")
    ear_hip_vertical_tilt_deg: float = Field(..., description="귀-골반 수직 기울기 (°)")
    shoulder_line_horizontal_tilt_deg: float = Field(..., description="어깨 선 수평 기울기 (°)")
    hip_line_horizontal_tilt_deg: float = Field(..., description="골반 선 수평 기울기 (°)")


class Changes(BaseModel):
    shoulder_height_diff_px: Optional[float] = Field(None, description="어깨 높이 차이 변화량 (px), 이전 업로드와 비교")
    hip_height_diff_px: Optional[float] = Field(None, description="골반 높이 차이 변화량 (px), 이전 업로드와 비교")
    torso_vertical_tilt_deg: Optional[float] = Field(None, description="몸통 수직 기울기 변화량 (°), 이전 업로드와 비교")
    ear_hip_vertical_tilt_deg: Optional[float] = Field(None, description="귀-골반 수직 기울기 변화량 (°), 이전 업로드와 비교")
    shoulder_line_horizontal_tilt_deg: Optional[float] = Field(None, description="어깨 선 수평 기울기 변화량 (°), 이전 업로드와 비교")
    hip_line_horizontal_tilt_deg: Optional[float] = Field(None, description="골반 선 수평 기울기 변화량 (°), 이전 업로드와 비교")


class HistoryItem(BaseModel):
    feature_id: int = Field(..., description="Feature 레코드 고유 ID")
    shoulder_height_diff_px: float = Field(..., description="해당 시점의 어깨 높이 차이 (px)")
    hip_height_diff_px: float = Field(..., description="해당 시점의 골반 높이 차이 (px)")
    torso_vertical_tilt_deg: float = Field(..., description="해당 시점의 몸통 수직 기울기 (°)")
    ear_hip_vertical_tilt_deg: float = Field(..., description="해당 시점의 귀-골반 수직 기울기 (°)")
    shoulder_line_horizontal_tilt_deg: float = Field(..., description="해당 시점의 어깨 선 수평 기울기 (°)")
    hip_line_horizontal_tilt_deg: float = Field(..., description="해당 시점의 골반 선 수평 기울기 (°)")
    composite_score: Optional[float] = Field(None, description="해당 시점의 종합점수 (각 지표 평균)")
    created_at: str = Field(..., description="레코드가 생성된 날짜-시간 (ISO8601, KST)")


class Recommendation(BaseModel):
    exercise: str = Field(..., description="추천 운동 이름")
    probability: float = Field(..., description="추천 확률 (%)")


class VideoItem(BaseModel):
    exercise: str = Field(..., description="관련 운동 이름")
    video_title: str = Field(..., description="영상 제목")
    video_desc: str = Field(..., description="영상 설명")
    video_url: str = Field(..., description="영상 재생 URL")
    thumbnail_url: Optional[str] = Field(None, description="영상 썸네일 URL")


class UploadResponse(BaseModel):
    message: str = Field(..., description="응답 메시지")
    user: UserInfo = Field(..., description="사용자 정보")
    visuals: Visuals = Field(..., description="시각화 이미지 URL 모음")
    metrics: Metrics = Field(..., description="계산된 6개 지표값")
    composite_score_current: Optional[float] = Field(None, description="이번 업로드의 종합점수 (각 지표 평균)")
    composite_percentile: Optional[float] = Field(None, description="종합점수 퍼센타일 (%)")
    composite_mean: Optional[float] = Field(None, description="동일 성별·연령 그룹의 종합점수 평균")
    composite_std: Optional[float] = Field(None, description="동일 성별·연령 그룹의 종합점수 표준편차")
    composite_min: Optional[float] = Field( None, description="같은 그룹(성별·연령) 내의 전체 종합점수 중 최소값")
    composite_max: Optional[float] = Field(None, description="같은 그룹(성별·연령) 내의 전체 종합점수 중 최대값")
    last_upload_date: Optional[str] = Field(None, description="마지막 업로드된 날짜-시간 (ISO8601, KST)")
    days_since: Optional[int] = Field(None, description="마지막 업로드 이후 경과 일수")
    changes: Changes = Field(..., description="지표별 변화량")
    history: List[HistoryItem] = Field(..., description="최근 5회 업로드 이력 (오름차순 정렬된 리스트)")
    recommendations: List[Recommendation] = Field(..., description="추천 운동 리스트 (상위 3개)")
    videos: List[VideoItem] = Field(..., description="추천 영상 리스트 (최대 3개씩)")
    report: str = Field(..., description="LLM이 생성한 종합 리포트 (한글 텍스트)")