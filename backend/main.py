from fastapi import FastAPI, Depends, UploadFile, File
from sqlalchemy.orm import Session
from database import SessionLocal, Base, engine
from models import Job
from crud import get_all_jobs
from nlp import summarize_resume
from rag import build_faiss_index, search_similar_jobs
from llm import generate_recommendation

Base.metadata.create_all(bind=engine)

app = FastAPI()

# DB 연결 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """이력서 업로드 및 일자리 추천"""
    content = await file.read()
    resume_text = content.decode("utf-8")

    # Step 1: NLP 요약
    summary = summarize_resume(resume_text)

    # Step 2: RAG를 이용한 유사 일자리 검색
    all_jobs = get_all_jobs(db)
    job_descriptions = [job.description for job in all_jobs]
    job_titles = [job.title for job in all_jobs]

    build_faiss_index(job_descriptions)  # FAISS 인덱스 생성
    indices, _ = search_similar_jobs(summary)

    recommended_titles = [job_titles[i] for i in indices]
    recommended_descriptions = [job_descriptions[i] for i in indices]

    # Step 3: LLM 기반 추천 결과 생성
    recommendation = generate_recommendation(resume_text, recommended_titles, recommended_descriptions)

    return {
        "summary": summary,
        "recommended_titles": recommended_titles,
        "recommended_descriptions": recommended_descriptions,
        "recommendation_reasoning": recommendation
    }
