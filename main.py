import io
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
import torch
import whisper
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

# ==========================
# НАСТРОЙКИ
# ==========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openrouter/auto")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_SYSTEM_PROMPT = os.getenv(
    "OPENAI_SYSTEM_PROMPT",
    "Ты дружелюбный голосовой ассистент. Отвечай коротко и по делу."
)

# Язык для STT и TTS
LANGUAGE = "ru"

# Голос Silero TTS (используем доступный в модели)
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "kseniya_v2")

# ==========================
# ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================

torch.set_num_threads(4)
_DEVICE_TTS = "cpu"
_stt_model = None
_tts_model = None
_tts_sample_rate = None


def _get_stt_model():
    global _stt_model
    if _stt_model is None:
        print("Загружаю модель распознавания речи (whisper)...")
        _stt_model = whisper.load_model("small")
    return _stt_model


def _ensure_quantized_engine():
    supported_qengines = getattr(torch.backends.quantized, "supported_engines", [])
    if supported_qengines:
        if "qnnpack" in supported_qengines:
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = supported_qengines[0]


def _get_tts_model():
    global _tts_model, _tts_sample_rate
    if _tts_model is None:
        print("Загружаю модель TTS (Silero)...")
        _ensure_quantized_engine()
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=TTS_SPEAKER,
            trust_repo=True
        )
        _tts_model = model.to(_DEVICE_TTS)
        _tts_sample_rate = getattr(_tts_model, "sample_rate", 16000)
    return _tts_model, _tts_sample_rate


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def speech_to_text(audio_path: str, language: Optional[str] = None) -> str:
    """
    Прогоняем аудио через whisper и возвращаем распознанный текст.
    """
    model = _get_stt_model()
    result = model.transcribe(audio_path, language=language)
    text = result["text"].strip()
    print(f"[STT] -> {text}")
    return text


def ask_llm(prompt: str) -> str:
    """
    Отправляем текст в OpenAI и получаем ответ.
    """
    print(f"[LLM запрос] {prompt}")

    if not OPENAI_API_KEY:
        msg = "Не задан OPENAI_API_KEY. Добавьте ключ в переменные окружения."
        print(f"[LLM ошибка] {msg}")
        return msg

    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("Пустой ответ от OpenAI")
        answer = choices[0]["message"]["content"].strip()
        print(f"[LLM ответ] {answer}")
        return answer
    except Exception as e:
        print(f"[LLM ошибка] {e}")
        return f"Произошла ошибка при обращении к внешнему API: {str(e)}"


def text_to_speech(text: str, sample_rate: Optional[int] = None) -> io.BytesIO:
    """
    Прогоняем текст через Silero TTS и возвращаем байты WAV в памяти.
    """
    model, default_sr = _get_tts_model()
    sr = sample_rate or default_sr
    with torch.no_grad():
        # Для v2+ моделей speaker уже задан при загрузке
        audio = model.apply_tts(
            texts=[text],
            sample_rate=sr
        )

    if isinstance(audio, (list, tuple)):
        audio_tensor = audio[0]
    else:
        audio_tensor = audio

    audio_np = audio_tensor.cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # audio -> WAV (PCM16) в буфер
    buffer = io.BytesIO()
    sf.write(buffer, audio_int16, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


# ==========================
# FASTAPI ПРИЛОЖЕНИЕ
# ==========================

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "static" / "index.html"


app = FastAPI(title="Local Voice Assistant on Mac")


@app.get("/")
async def index():
    return FileResponse(
        INDEX_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-store"}
    )


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    """
    Принимает аудио-файл (wav/ogg/m4a и т.п.), возвращает аудио-ответ wav.
    """
    # Сохраняем входной файл во временную директорию
    suffix = ""
    if file.filename:
        suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        guessed = mimetypes.guess_extension(file.content_type or "")
        suffix = guessed or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        # 1. STT: аудио -> текст
        try:
            text = speech_to_text(tmp_path, language=LANGUAGE)
            if not text:
                answer_text = "Я не расслышал ваш вопрос. Попробуйте еще раз."
            else:
                # 2. LLM: текст -> ответ
                answer_text = ask_llm(text)
        except Exception as e:
            print(f"[Ошибка обработки] {e}")
            answer_text = "Произошла ошибка при обработке вашего запроса."

        # 3. TTS: ответ -> аудио (WAV)
        audio_buffer = text_to_speech(answer_text)

        # Возвращаем аудио как StreamingResponse
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="reply.wav"'
            }
        )
    finally:
        # Удаляем временный файл
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # Запуск сервера: python main.py
    host = "0.0.0.0"
    port = 8000
    print(f"Сервер запущен: http://localhost:{port} (доступен также по http://{host}:{port})")
    uvicorn.run("main:app", host=host, port=port, reload=True)
