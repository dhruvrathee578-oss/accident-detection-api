from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels for classification
labels = [
    # ACCIDENT
    "car collision accident",
    "highway crash accident",
    "motorcycle crash accident",
    "truck rollover accident",
    "bus crash accident",
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

    # NON-ACCIDENT
    "normal traffic road",
    "parked vehicles",
    "traffic jam",
    "night driving normal",
    "bonfire",
    "kitchen cooking flame",
    "fireworks celebration",
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
    image = Image.open(file.file).convert("RGB")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    predicted_index = probs.argmax().item()
    predicted_label = labels[predicted_index]

    if "accident" in predicted_label or "crash" in predicted_label or "fallen" in predicted_label:
        result = "ACCIDENT"
    else:
        result = "NON-ACCIDENT"

    return JSONResponse({
        "prediction": result,
        "label": predicted_label
    })
