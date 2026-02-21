from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")  # Render Environment Variable
API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ✅ FULL LABELS (Accident + Non-Accident) as you asked
labels = [
    # ---------------- ACCIDENT (Road/Vehicle) ----------------
    "car-to-car collision accident",
    "highway pile-up accident",
    "motorcycle crash accident",
    "truck rollover accident",
    "bus accident",
    "pedestrian hit by vehicle accident",
    "rear-end collision accident",
    "head-on collision accident",
    "side-impact (T-bone) accident",
    "vehicle overturn accident",

    # ---------------- ACCIDENT (Fire & Explosion) ----------------
    "building fire accident",
    "vehicle fire accident",
    "electrical fire accident",
    "industrial explosion accident",
    "gas cylinder blast accident",
    "chemical explosion accident",
    "forest fire accident",

    # ---------------- ACCIDENT (Industrial/Construction) ----------------
    "worker fall from height accident",
    "machine injury accident",
    "scaffold collapse accident",
    "crane accident",
    "heavy object fall accident",
    "electrical shock accident",
    "factory fire accident",

    # ---------------- ACCIDENT (Human Fall/Medical) ----------------
    "slip and fall accident",
    "elderly fall accident",
    "sudden collapse accident",
    "fainting accident",
    "person lying unconscious accident",

    # ---------------- ACCIDENT (Water) ----------------
    "boat collision accident",
    "drowning incident accident",
    "vehicle submerged in water accident",
    "flood-related accident",

    # ================= NON-ACCIDENT =================

    # ---------------- Non-Accident (Normal Road) ----------------
    "normal traffic",
    "vehicles parked",
    "traffic jam (not accident)",
    "speed breakers on road",
    "night driving normal road",

    # ---------------- Non-Accident (Fire-like but normal) ----------------
    "bonfire (controlled fire, not accident)",
    "kitchen flame (cooking, not accident)",
    "fireworks (celebration, not accident)",
    "controlled industrial flame (not accident)",

    # ---------------- Non-Accident (Normal Human Activities) ----------------
    "sitting on ground normally",
    "sleeping person",
    "yoga pose",
    "worker bending normally",
    "playing sports",

    # ---------------- Non-Accident (Industrial Normal) ----------------
    "normal machine operation",
    "safe construction work",
    "crane lifting normally",
    "workers with safety gear"
]


@app.get("/")
def home():
    return {"status": "ok", "message": "Accident Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not HF_TOKEN:
        return JSONResponse(status_code=500, content={"error": "HF_TOKEN not set in environment variables"})

    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": image_b64,
        "parameters": {
            "candidate_labels": labels
        }
    }

    r = requests.post(API_URL, headers=headers, json=payload)

    try:
        out = r.json()
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Invalid response from HuggingFace", "raw": r.text})

    # ✅ Output clean ACCIDENT / NON-ACCIDENT
    # HuggingFace usually returns: [{"label": "...", "score": 0.xx}, ...] OR dict
    top_label = None
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "label" in out[0]:
        top_label = out[0]["label"]
    elif isinstance(out, dict) and "label" in out:
        top_label = out["label"]

    if not top_label:
        return JSONResponse(content={"raw_result": out})

    if "accident" in top_label or "crash" in top_label or "collision" in top_label or "blast" in top_label:
        final_pred = "ACCIDENT"
    else:
        final_pred = "NON-ACCIDENT"

    return JSONResponse(content={
        "prediction": final_pred,
        "top_label": top_label,
        "raw_result": out
    })
