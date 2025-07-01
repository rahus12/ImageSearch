from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import faiss
import numpy as np
import shutil
import os

app = FastAPI()

# --- Load existing index and labels ---
INDEX_PATH = "face_index.faiss"
LABELS_PATH = "face_labels.npy"
IMAGE_PATHS_PATH = "image_paths.npy"

index = faiss.read_index(INDEX_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
image_paths = np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()

model = DeepFace.build_model("Facenet")

# --- Utility Functions ---
def get_embedding(image_path):
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        model=model,
        enforce_detection=False
    )[0]["embedding"]
    return np.array([embedding], dtype="float32")

def find_similar(embedding, k=1):
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, k)
    return distances[0], indices[0]

def save_to_disk(temp_file: UploadFile, label: str = None):
    os.makedirs("saved_faces", exist_ok=True)
    filename = temp_file.filename.replace(" ", "_")
    save_name = f"{label}_{filename}" if label else filename
    save_path = os.path.join("saved_faces", save_name)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(temp_file.file, buffer)
    return save_path

# --- Endpoint: Search face ---
@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    img_path = save_to_disk(file)
    embedding = get_embedding(img_path)
    distances, indices = find_similar(embedding, k=1)

    if distances[0] < 0.7:
        return {"match": None, "message": "No confident match found."}

    return {
        "match": labels[indices[0]],
        "similarity": float(distances[0]),
        "path": image_paths[indices[0]]
    }

# --- Endpoint: Add new face ---
class AddFaceRequest(BaseModel):
    label: str

@app.post("/add")
async def add_face(file: UploadFile = File(...), metadata: AddFaceRequest = None):
    img_path = save_to_disk(file, label=metadata.label)
    embedding = get_embedding(img_path)
    distances, indices = find_similar(embedding, k=1)

    if distances[0] >= 0.95:
        raise HTTPException(status_code=400, detail="A similar face already exists in the database. Not added.")

    # Add to FAISS
    faiss.normalize_L2(embedding)
    index.add(embedding)

    # Update metadata
    labels.append(metadata.label)
    image_paths.append(img_path)

    # Save updated index and metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(LABELS_PATH, np.array(labels))
    np.save(IMAGE_PATHS_PATH, np.array(image_paths))

    return {"message": f"New face for '{metadata.label}' added successfully."}