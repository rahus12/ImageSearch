# Scalable Face Recognition API with DeepFace, FAISS & FastAPI

## 📌 Overview
This project started as a simple university assignment for face classification but quickly evolved into a **scalable, production-ready face recognition API**.

Instead of training a fixed classification model, this system uses **embeddings + vector search** for flexibility and speed:
- No retraining needed when adding new faces
- Handles large datasets efficiently with **FAISS**
- Provides a clean REST API using **FastAPI**
- Prevents duplicate face entries using similarity thresholds

---

## 🚀 Features
- **Face Search**: Given an image, find the closest match from the database.
- **Add New Face**: Upload a new face with a label. Rejects if similarity ≥ 0.95 to avoid duplicates.
- **Scalable**: Designed to handle thousands to millions of faces with low latency.
- **Persistent Storage**: FAISS index, labels, and image paths saved for reuse.
- **API-First Design**: Can integrate easily into any app or service.

---

## 🛠 Tech Stack
- **[DeepFace](https://github.com/serengil/deepface)** (Facenet backend) – for generating face embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** – for efficient similarity search
- **[FastAPI](https://fastapi.tiangolo.com/)** – for building REST endpoints
- **NumPy** – for metadata storage
- **Python 3.8+**

---

## To run
### Clone the repository
- git clone https://github.com/rahus12/face-recognition-api.git
- cd face-recognition-api
- Run model.py to create the vector store

### Run the app
- uvicorn main:app --reload

