import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.ocr_engine import OCREngine

app = FastAPI(title="Interactive OCR Hub", description="A FastAPI + HTML/JS OCR Web Dashboard using KNN and Keras CNN")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = OCREngine()

# Ensure frontend folders exist before mounting
frontend_path = "c:/Users/chras/Desktop/IMP/Projects/frontend"
os.makedirs(os.path.join(frontend_path, "css"), exist_ok=True)
os.makedirs(os.path.join(frontend_path, "js"), exist_ok=True)

# Mount static folders
app.mount("/css", StaticFiles(directory=os.path.join(frontend_path, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(frontend_path, "js")), name="js")
app.mount("/dataset", StaticFiles(directory="c:/Users/chras/Desktop/IMP/Projects/backend/dataset"), name="dataset")

@app.get("/")
def read_root():
    index_file = os.path.join(frontend_path, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h1>OCR Frontend index.html not built yet. Please write it first.</h1>")
    return FileResponse(index_file)

@app.post("/api/upload-template")
async def upload_template(label_char: str = Form(...), file: UploadFile = File(...)):
    """
    Uploads a grid template of character class, segments it, and stores the isolated components.
    """
    label_char = label_char.strip().lower()
    if len(label_char) != 1 or not label_char.isalpha():
        raise HTTPException(status_code=400, detail="Invalid character label. Must be a single alphabet character.")
    
    if label_char not in engine.char_map:
        raise HTTPException(status_code=400, detail=f"Label '{label_char}' is not in allowed mapping list: {list(engine.char_map.keys())}")
    
    try:
        content = await file.read()
        num_saved = engine.store_training_template(label_char, content)
        return {
            "success": True,
            "message": f"Successfully segmented and saved {num_saved} components for character '{label_char}'."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
def train_models():
    """
    Triggers feature extraction, KNN training, and Keras CNN model training.
    """
    try:
        res = engine.train_models()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_image(method: str = "knn", file: UploadFile = File(...)):
    """
    Uploads a test bitmap image, segments character blobs, and evaluates them.
    Method must be either 'knn' or 'cnn'.
    """
    method = method.lower()
    if method not in ["knn", "cnn"]:
        raise HTTPException(status_code=400, detail="Method must be either 'knn' or 'cnn'.")
    
    try:
        content = await file.read()
        res = engine.predict_image(content, method=method)
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def get_status():
    """
    Gets model trained status, template image count per category, etc.
    """
    try:
        status = engine.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
