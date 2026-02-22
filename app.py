from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

CAPTION_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
ZSC_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

@app.get("/")
def home():
    return {"status": "ok", "message": "Accident Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not HF_TOKEN:
        return JSONResponse(status_code=500, content={"error": "HF_TOKEN not set"})

    image_bytes = await file.read()

    # A) Image -> Caption
    cap = requests.post(CAPTION_URL, headers=HEADERS, data=image_bytes, timeout=60)
    if cap.status_code == 503:
        return JSONResponse(status_code=503, content={"error": "Caption model loading. Try again in 20-40 seconds."})
    cap_json = cap.json()
    if isinstance(cap_json, dict) and cap_json.get("error"):
        return JSONResponse(status_code=500, content={"error": cap_json["error"]})
    try:
        caption = cap_json[0]["generated_text"]
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Caption parse failed", "raw": cap_json})

    # B) Caption -> ACCIDENT / NON-ACCIDENT
    payload = {
        "inputs": caption,
        "parameters": {"candidate_labels": ["ACCIDENT", "NON-ACCIDENT"]}
    }
    zsc = requests.post(ZSC_URL, headers=HEADERS, json=payload, timeout=60)
    if zsc.status_code == 503:
        return JSONResponse(status_code=503, content={"error": "Classifier loading. Try again in 20-40 seconds."})
    zsc_json = zsc.json()
    if isinstance(zsc_json, dict) and zsc_json.get("error"):
        return JSONResponse(status_code=500, content={"error": zsc_json["error"]})

    try:
        prediction = zsc_json["labels"][0]
        confidence = float(zsc_json["scores"][0])
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Classification parse failed", "raw": zsc_json})

    return JSONResponse(content={
        "prediction": prediction,   # ACCIDENT / NON-ACCIDENT
        "confidence": round(confidence, 4),
        "caption": caption
    })
