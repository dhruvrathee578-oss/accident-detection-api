from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

labels = [
    # ACCIDENT
    "car collision accident",
    "highway crash accident",
    "motorcycle crash accident",
    "truck rollover accident",
    "bus accident",
    "pedestrian accident",
    "building fire accident",
    "vehicle fire accident",
    "industrial explosion accident",
    "gas cylinder blast accident",
    "worker fall accident",
    "scaffold collapse accident",
    "crane accident",
    "electrical shock accident",
    "factory fire accident",
    "human fallen unconscious accident",
    "drowning accident",
    "boat collision accident",
    "vehicle submerged in water accident",
    "flood disaster accident",

    # NON ACCIDENT
    "normal traffic road",
    "parked vehicles",
    "traffic jam",
    "night driving normal",
    "bonfire",
    "kitchen cooking flame",
    "fireworks",
    "controlled industrial flame",
    "people sitting normally",
    "person sleeping",
    "yoga pose",
    "worker bending safely",
    "playing sports",
    "normal construction work",
    "crane lifting safely",
    "workers wearing safety gear",
    "normal machine operation"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    response = requests.post(
        API_URL,
        headers=headers,
        data=image_bytes
    )

    result = response.json()

    return JSONResponse(content=result)
