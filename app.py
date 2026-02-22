from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, base64, requests, json, re

app = FastAPI()

# Render Environment Variable: GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Use a model that exists in your list (you shared gemini-2.0-flash, gemini-2.5-flash, etc.)
MODEL_NAME = "gemini-2.0-flash"   # ✅ Free-tier friendly & stable
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

@app.get("/")
def home():
    return {"status": "ok", "message": "Accident Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "GEMINI_API_KEY not set in Render Environment Variables"})

    # Read image
    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    # Strict prompt
    prompt = (
        "You are a strict accident detection classifier.\n"
        "Task: Look at the image and classify ONLY one:\n"
        "- accident\n"
        "- non-accident\n\n"
        "Return ONLY valid JSON, no extra text.\n"
        "JSON format must be exactly: {\"label\":\"accident\"} OR {\"label\":\"non-accident\"}\n"
        "If unsure, choose the most likely label.\n"
    )

    # ✅ Correct Gemini payload (camelCase)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {          # ✅ NOT inline_data
                            "mimeType": mime,    # ✅ NOT mime_type
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 20,
            "responseMimeType": "application/json"  # ✅ forces JSON output
        }
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Connection error", "details": str(e)})

    # If Gemini returns error
    if resp.status_code != 200:
        try:
            return JSONResponse(status_code=resp.status_code, content={"error": "Gemini API error", "details": resp.json()})
        except Exception:
            return JSONResponse(status_code=resp.status_code, content={"error": "Gemini API error", "details": resp.text})

    data = resp.json()

    # Extract text
    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Unexpected Gemini response", "raw": data})

    # Parse JSON safely
    # Sometimes model returns JSON string with whitespace/newlines
    try:
        result_json = json.loads(text_out)
    except Exception:
        # fallback: try to extract JSON object from text
        m = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
        if not m:
            return JSONResponse(status_code=500, content={"error": "Could not parse model JSON", "raw_text": text_out})
        result_json = json.loads(m.group(0))

    label = (result_json.get("label") or "").strip().lower()
    if label not in ["accident", "non-accident"]:
        return JSONResponse(status_code=500, content={"error": "Invalid label from model", "model_output": result_json})

    # ✅ Final output (professional)
    return JSONResponse(content={"label": label})
