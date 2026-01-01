from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
from process_image import process_image
import numpy as np
import os

app = FastAPI()
app.mount("/output", StaticFiles(directory="output"), name="output")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
RESULT_IMAGE_PATH = os.path.join(TEMP_DIR, "result.png")

@app.post("/generate_image")
async def generate_image(file: UploadFile = File(...), beast_id: str = Form(...)):
    contents = await file.read()

    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    result_img = process_image(img, beast_id)

    cv2.imwrite(RESULT_IMAGE_PATH, result_img)

    print("WRITE RESULT:", cv2.imwrite(RESULT_IMAGE_PATH, result_img))
    print("FILE EXISTS:", os.path.exists(RESULT_IMAGE_PATH))
    print("ABS PATH:", os.path.abspath(RESULT_IMAGE_PATH))


    return {"status": "ok"}

@app.get("/get_image/{image_name}")
def get_image(image_name: str):
    image_path = os.path.join("temp", image_name)

    if not os.path.exists(image_path):
        return {"error": "image not found"}

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=image_name
    )