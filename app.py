from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import base64
import requests

app = FastAPI()

# Render Environment Variable: GEMINI_API_KEY
# .strip() use kora hoyeche jate kono extra space na thake
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Model URL ta ekta single line e rakha bhalo jate kono newline ba space error na hoy
# Version v1beta e rakha hoyeche karon flash model eitei bhalo kaj kore
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

@app.get("/")
def home():
    return {"status": "ok", "message": "Accident Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY not set in Render Environment Variables"}
        )

    # Image processing
    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prompt structure
    prompt = (
        "You are an accident detection system.\n"
        "Look at the image and decide ONLY one label:\n"
        "1) accident\n"
        "2) non-accident\n\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- JSON format: {\"label\":\"accident\"} or {\"label\":\"non-accident\"}\n"
        "- No extra text.\n"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": file.content_type or "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50
        }
    }

    headers = {"Content-Type": "application/json"}

    # Request path e query parameter hisebe API key pass kora hochhe
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=60
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Connection Error", "details": str(e)}
        )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": "Gemini API error", "details": resp.json()}
        )

    data = resp.json()

    # Model theke response extract kora
    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        # JSON string theke extra whitespace ba newline remove kora
        clean_result = text_out.strip()
        return JSONResponse(content={"result": clean_result})
    except (KeyError, IndexError):
        return JSONResponse(
            status_code=500, 
            content={"error": "Unexpected Gemini response structure", "raw": data}
        )
