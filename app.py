from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import base64
import requests
import json

app = FastAPI()

# Render Environment Variable: GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ✅ Use a model that exists in your list (you showed gemini-2.5-flash etc.)
MODEL_NAME = "models/gemini-2.5-flash"

# v1beta endpoint (OK for generateContent)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent"

@app.get("/")
def home():
    return {"status": "ok", "message": "Accident Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) API key check
    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY not set in Render Environment Variables"}
        )

    # 2) Read image and convert to base64
    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # 3) Strong prompt (ONLY JSON output)
    prompt = (
        "You are an accident detection system.\n"
        "Decide ONLY ONE label for the image:\n"
        "1) accident\n"
        "2) non-accident\n\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Exactly this format: {\"label\":\"accident\"} OR {\"label\":\"non-accident\"}\n"
        "- No extra words, no explanation.\n"
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

    # 4) Call Gemini
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

    # 5) If Gemini returns error
    if resp.status_code != 200:
        # try to show json, otherwise show text
        try:
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": "Gemini API error", "details": resp.json()}
            )
        except:
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": "Gemini API error", "details": resp.text}
            )

    data = resp.json()

    # 6) Extract model output text
    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        clean_text = text_out.strip()

        # 7) Parse JSON output safely
        # Sometimes Gemini returns JSON as a string -> we convert it to dict
        try:
            result_obj = json.loads(clean_text)
        except:
            # If parsing fails, fallback: try to detect label from text
            lower = clean_text.lower()
            if "non-accident" in lower:
                result_obj = {"label": "non-accident"}
            elif "accident" in lower:
                result_obj = {"label": "accident"}
            else:
                result_obj = {"label": "unknown", "raw": clean_text}

        # 8) Return clean final output
        # Output will be {"label":"accident"} or {"label":"non-accident"}
        return JSONResponse(content=result_obj)

    except (KeyError, IndexError):
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Gemini response structure", "raw": data}
        )
