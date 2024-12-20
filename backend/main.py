from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Request , WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS
from langchain.schema import Document
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import base64


# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# Static 파일 및 템플릿 경로 설정
BASE_DIR = Path(__file__).resolve().parent
# app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지: 버튼 두 개 제공"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """이력서 업로드 페이지"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/summarize")
async def summarize_resume(request: Request, file: UploadFile = File(...)):
    # 업로드된 파일을 임시 위치에 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_file_name = tmp_file.name

    # 파일 형식에 따라 적절한 로더 선택
    if file.filename.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_name)
    elif file.filename.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_name)
    else:
        os.remove(tmp_file_name)
        return JSONResponse(content={"error": "지원하지 않는 파일 형식입니다."}, status_code=400)

    documents = loader.load()
    print(documents)
    # 임시 파일 삭제
    os.remove(tmp_file_name)

    # LLM 초기화 (API 키를 환경 변수에서 가져옴)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # 한 문장 요약을 위한 프롬프트 설정
    prompt_template = """다음 내용을 구직활동에 도움이 되는 핵심정보만 포함하여 한 문장으로 요약하세요:

{text}

요약:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 요약 체인 로드
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    # 요약 실행
    summary = chain.run(documents)

    # 결과를 프론트로 전달
    return templates.TemplateResponse("result.html", {"request": request, "summary": summary})


@app.get("/create", response_class=HTMLResponse)
async def create_page(request: Request):
    """이력서 작성 페이지"""
    return templates.TemplateResponse("create.html", {"request": request})


@app.post("/submit")
async def submit_resume(
    request: Request,
    name: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    contact: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    education: Optional[str] = Form(None),
    certifications: Optional[str] = Form(None),
    skills: Optional[str] = Form(None),
    desired_job: Optional[str] = Form(None),
    desired_location: Optional[str] = Form(None)
):
    """작성된 이력서를 요약"""
    # 이력서 텍스트 생성
    resume_text = f"""
    이름: {name or "정보 없음"}
    나이: {age or "정보 없음"}
    연락처: {contact or "정보 없음"}
    주요 경력: {experience or "정보 없음"}
    학력: {education or "정보 없음"}
    자격증: {certifications or "정보 없음"}
    기술: {skills or "정보 없음"}
    희망 직무: {desired_job or "정보 없음"}
    희망 근무지: {desired_location or "정보 없음"}
    """

    # `Document` 객체로 변환
    document = Document(page_content=resume_text)

    # LLM 초기화
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # 요약 프롬프트
    prompt_template = """다음 내용을 구직활동에 도움이 되는 핵심정보만 포함하여 한 문장으로 요약하세요:

{text}

요약:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    # `chain.invoke`를 사용하여 요약 수행
    summary = chain.invoke([document])

    return templates.TemplateResponse("result.html", {"request": request, "summary": summary})


# @app.post("/ask_question")
# async def ask_question(job_description: str = Form(...)):
#     """OpenAI를 사용하여 맞춤형 면접 질문 생성"""
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     job_description = '60세 이상 우대, 서울시립도서관에서 도서 대출 보조, 서가 정리 및 이용자 안내를 담당할 시간제 근로자를 모집합니다.'

#     if not openai_api_key:
#         return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

# @app.get("/ask_question")
# async def ask_question():
#     """
#     LangChain을 사용하여 고정된 job_description을 기반으로 면접 질문 생성
#     """
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

#     try:
#         # 고정된 Job Description
#         job_description = "60세 이상 우대, 서울시립도서관에서 도서 대출 보조, 서가 정리 및 이용자 안내를 담당할 시간제 근로자를 모집합니다."

#         # LLM 초기화
#         llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

#         # 프롬프트 템플릿 정의
#         prompt_template = PromptTemplate(
#             input_variables=["job_description"],
#             template="""
#             다음 채용공고를 기반으로 구직자의 역량을 평가할 수 있는 면접 질문 3개를 한국어로 생성하세요:
            
#             채용공고:
#             {job_description}

#             면접 질문:
#             """
#         )

#         # LangChain LLMChain 구성
#         chain = LLMChain(llm=llm, prompt=prompt_template)

#         # LangChain을 사용하여 질문 생성
#         generated_questions = chain.run({"job_description": job_description})
#         question_text = generated_questions.strip()

#         # TTS로 질문 음성 생성
#         tts = gTTS(text=question_text, lang="ko")
#         tts.save("question.mp3")

#         return FileResponse("question.mp3")

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/generate_questions")
async def generate_questions():
    """LangChain을 사용하여 고정된 job_description을 기반으로 면접 질문 생성"""
    global generated_questions
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    job_description = "60세 이상 우대, 서울시립도서관에서 도서 대출 보조, 서가 정리 및 이용자 안내를 담당할 시간제 근로자를 모집합니다."

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    prompt_template = PromptTemplate(
        input_variables=["job_description"],
        template="""
        다음 채용공고를 기반으로 구직자의 역량을 평가할 수 있는 면접 질문 3개를 한국어로 생성하세요:
        
        채용공고:
        {job_description}

        면접 질문:
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    generated_questions = chain.run({"job_description": job_description})
    generated_questions = [q.strip("- ") for q in generated_questions.strip().split("\n") if q.strip()]

    return {"questions": generated_questions}

@app.get("/tts/{question_index}", response_class=HTMLResponse)
async def tts_page(request: Request, question_index: int):
    """
    특정 질문의 텍스트와 음성 파일을 HTML 페이지로 렌더링
    """
    global generated_questions

    try:
        # 질문 유효성 확인
        if question_index < 0 or question_index >= len(generated_questions):
            return JSONResponse(content={"error": "Invalid question index"}, status_code=400)

        question_text = generated_questions[question_index]

        # TTS 음성 생성
        tts = gTTS(text=question_text, lang="ko")
        tts_file = f"static/question_{question_index}.mp3"
        tts.save(tts_file)

        # HTML 페이지 렌더링
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "question_text": question_text,
                "audio_file": f"/static/question_{question_index}.mp3",
                "next_index": question_index + 1 if question_index + 1 < len(generated_questions) else None,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# TensorFlow Lite 모델 경로
TFLITE_MODEL_PATH = "/Users/eonseon/senior-job-recommendation/backend/model_unquant.tflite"

# TFLite 모델 로드
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    class_names = ["confident", "unconfident"]  # Teachable Machine에서 설정한 클래스 이름
    print("TensorFlow Lite 모델 로드 성공")
except Exception as e:
    print(f"TensorFlow Lite 모델 로드 실패: {e}")
    interpreter = None

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket 엔드포인트: 클라이언트에서 프레임을 실시간으로 받아 분석
    """
    if interpreter is None:
        await websocket.close(code=1000)
        print("모델이 로드되지 않았습니다.")
        return

    await websocket.accept()
    try:
        while True:
            # 클라이언트로부터 Base64로 인코딩된 이미지 데이터 수신
            data = await websocket.receive_text()
            image_data = base64.b64decode(data)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_json({"error": "이미지 디코딩 실패"})
                continue

            # 이미지를 모델 입력 크기로 조정
            img = cv2.resize(img, (224, 224))
            input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

            # 모델 예측
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            # 예측 결과 추출
            confidence = float(np.max(predictions))  # 가장 높은 확률 값
            label = class_names[np.argmax(predictions)]  # 가장 높은 확률에 해당하는 클래스 이름

            # 결과를 클라이언트로 전송
            await websocket.send_json({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"WebSocket 에러: {e}")
    finally:
        await websocket.close()

@app.post("/classify")
async def classify_webcam_image(file: UploadFile = File(...)):
    """
    TensorFlow Lite 모델로 웹캠 이미지를 분류
    """
    if interpreter is None:
        print("모델 로드 실패")
        return JSONResponse(content={"error": "TensorFlow Lite 모델이 로드되지 않았습니다."}, status_code=500)

    try:
        # 이미지를 읽고 처리
        print("이미지 읽기 시작")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("이미지 디코딩 완료")

        if img is None:
            print("이미지 디코딩 실패")
            return JSONResponse(content={"error": "이미지를 디코딩할 수 없습니다."}, status_code=400)

        # 이미지를 모델 입력 크기로 조정
        img = cv2.resize(img, (224, 224))  # Teachable Machine 모델의 입력 크기
        print("이미지 리사이즈 완료")

        # 이미지를 정규화하고 배치 차원 추가
        input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        print("이미지 정규화 완료")

        # 모델 예측
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # 예측 결과 추출
        confidence = float(np.max(predictions))  # 가장 높은 확률 값
        label = class_names[np.argmax(predictions)]  # 가장 높은 확률에 해당하는 클래스 이름

        print(f"모델 예측 결과: {predictions}, 라벨: {label}, 신뢰도: {confidence}")

        return JSONResponse({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"예외 발생: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# @app.post("/analyze_feedback")
# async def analyze_feedback(file: UploadFile = File(...)):
#     """Teachable Machine 모델을 사용하여 자세 분석 및 피드백 생성"""
#     try:
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (224, 224))
#         img = np.expand_dims(img, axis=0) / 255.0

#         predictions = model.predict(img)
#         label = class_names[np.argmax(predictions)]

#         feedback = {
#             "Confident": "Great! You look confident and professional.",
#             "Neutral": "Your posture is neutral. Try to show more enthusiasm.",
#             "Poor Posture": "Your posture needs improvement. Sit up straight and maintain eye contact."
#         }.get(label, "Unknown feedback.")

#         tts = gTTS(text=feedback, lang="en")
#         tts.save("feedback.mp3")

#         return FileResponse("feedback.mp3")

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)