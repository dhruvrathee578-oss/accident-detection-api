from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import base64
import requests

app = FastAPI()

# Render Environment Variable: GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini endpoint (Vision capable)
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

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

    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

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
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": file.content_type or "image/jpeg",
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

    resp = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload,
        timeout=60
    )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={"error": "Gemini API error", "status_code": resp.status_code, "details": resp.text}
        )

    data = resp.json()

    # Extract model text
    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Unexpected Gemini response", "raw": data})

    # Return exactly what model produced (should be JSON string)
    return JSONResponse(content={"result": text_out})
