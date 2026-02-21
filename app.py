from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

# Hugging Face Token (Render Environment Variable এ সেট করবে)
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ✅ তোমার দেওয়া সব টাইপ কভার করা labels
labels = [
    # ---------------- ACCIDENT: Road / Vehicle ----------------
    "car-to-car collision accident",
    "highway pile-up accident",
    "motorcycle crash accident",
    "truck rollover accident",
    "bus accident",
    "pedestrian hit by vehicle accident",
    "rear-end collision accident",
    "head-on collision accident",
    "side-impact t-bone accident",
    "vehicle overturn accident",

    # ---------------- ACCIDENT: Fire & Explosion ----------------
    "building fire accident",
    "vehicle fire accident",
    "electrical fire accident",
    "industrial explosion accident",
    "gas cylinder blast accident",
    "chemical explosion accident",
    "forest fire accident",

    # ---------------- ACCIDENT: Industrial / Construction ----------------
    "worker fall from height accident",
    "machine injury accident",
    "scaffold collapse accident",
    "crane accident",
    "heavy object fall accident",
    "electrical shock accident",
    "factory fire accident",

    # ---------------- ACCIDENT: Human Fall / Medical ----------------
    "slip and fall accident",
    "elderly fall accident",
    "sudden collapse accident",
    "fainting accident",
    "person lying unconscious accident",

    # ---------------- ACCIDENT: Water Related ----------------
    "boat collision accident",
    "drowning incident accident",
    "vehicle submerged in water accident",
    "flood-related accident",

    # ================= NON-ACCIDENT (very important) =================
    # Normal Road Scenes
    "normal traffic on road",
    "vehicles parked normally",
    "traffic jam not accident",
    "speed breaker on road",
    "night driving normal road",

    # Normal Fire-like (Not Accident)
    "bonfire controlled",
    "kitchen flame cooking",
    "fireworks celebration",
    "controlled industrial flame",

    # Normal Human Activities
    "person sitting on ground normal",
    "person sleeping normal",
    "yoga pose normal",
    "worker bending normal",
    "playing sports normal",

    # Industrial Non-Accident
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
    # ✅ Token check
    if not HF_TOKEN:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not set in Render Environment Variables"}
        )

    image_bytes = await file.read()

    # ✅ HuggingFace inference API call
    payload = {"inputs": image_bytes}

    resp = requests.post(API_URL, headers=headers, data=image_bytes)

    # If model is loading
    if resp.status_code == 503:
        return JSONResponse(status_code=503, content={"error": "Model loading. Try again in 20-40 seconds."})

    result = resp.json()

    # HuggingFace sometimes returns error dict
    if isinstance(result, dict) and result.get("error"):
        return JSONResponse(status_code=500, content={"error": result.get("error")})

    # CLIP output format varies; we do our own scoring using labels via zero-shot style:
    # For simplicity, we use the raw output if provided, otherwise return response.
    return JSONResponse(content={"hf_response": result, "note": "CLIP output format may vary. This API confirms connection."})
