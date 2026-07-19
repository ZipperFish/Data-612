"""
FastAPI service for the wine-food pairing recommender.

Loads model_artifacts.joblib (produced by train_model.py) either from local
disk or from Azure Blob Storage (set MODEL_SOURCE=blob and the related env
vars below), and exposes endpoints to predict a pairing score and to get
top-N recommendations.
"""
import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")  # "local" or "blob"
LOCAL_ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "model_artifacts.joblib")

# Blob storage settings (only used if MODEL_SOURCE=blob)
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER", "model-artifacts")
BLOB_NAME = os.environ.get("BLOB_NAME", "model_artifacts.joblib")

app = FastAPI(title="Wine & Food Pairing Recommender", version="1.0")

_artifacts = None


def load_artifacts():
    """Load model artifacts from local disk or Azure Blob Storage."""
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    if MODEL_SOURCE == "blob":
        from azure.storage.blob import BlobServiceClient
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")
        client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = client.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_NAME)
        local_tmp = "/tmp/" + BLOB_NAME
        with open(local_tmp, "wb") as f:
            f.write(blob_client.download_blob().readall())
        _artifacts = joblib.load(local_tmp)
    else:
        _artifacts = joblib.load(LOCAL_ARTIFACT_PATH)

    return _artifacts


@app.on_event("startup")
def startup_event():
    load_artifacts()


class PredictRequest(BaseModel):
    food_item: str
    wine_type: str


class PredictResponse(BaseModel):
    food_item: str
    wine_type: str
    predicted_score: float


class RecommendResponse(BaseModel):
    query_type: str
    query_value: str
    recommendations: list


def _predict_one(a, food_item, wine_type):
    if food_item not in a["food_idx"] or wine_type not in a["wine_idx"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown food_item or wine_type. "
                    f"Known foods/wines are listed at /foods and /wines.",
        )
    fi = a["food_idx"][food_item]
    wi = a["wine_idx"][wine_type]
    pred = a["mu"] + a["b_row"][fi] + a["b_col"][wi] + np.dot(a["P"][fi], a["Q"][wi])
    return float(np.clip(pred, 1, 5))


@app.get("/")
def root():
    return {
        "service": "Wine & Food Pairing Recommender",
        "endpoints": ["/predict", "/recommend/wines-for-food", "/recommend/foods-for-wine", "/foods", "/wines", "/health"],
    }


@app.get("/health")
def health():
    a = load_artifacts()
    return {"status": "ok", "test_rmse": a["test_rmse"], "test_mae": a["test_mae"]}


@app.get("/foods")
def list_foods():
    a = load_artifacts()
    return {"foods": a["food_items_sorted"]}


@app.get("/wines")
def list_wines():
    a = load_artifacts()
    return {"wines": a["wine_types_sorted"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    a = load_artifacts()
    score = _predict_one(a, req.food_item, req.wine_type)
    return PredictResponse(food_item=req.food_item, wine_type=req.wine_type, predicted_score=score)


@app.get("/recommend/wines-for-food", response_model=RecommendResponse)
def recommend_wines_for_food(food_item: str, top_n: int = 5):
    a = load_artifacts()
    if food_item not in a["food_idx"]:
        raise HTTPException(status_code=404, detail=f"Unknown food_item '{food_item}'. See /foods.")

    scored = a["scored_by_food"].get(food_item, set())
    candidates = [w for w in a["wine_types_sorted"] if w not in scored]
    preds = [(w, _predict_one(a, food_item, w)) for w in candidates]
    preds.sort(key=lambda x: -x[1])
    top = preds[:top_n]

    return RecommendResponse(
        query_type="wines_for_food",
        query_value=food_item,
        recommendations=[{"wine_type": w, "predicted_score": s} for w, s in top],
    )


@app.get("/recommend/foods-for-wine", response_model=RecommendResponse)
def recommend_foods_for_wine(wine_type: str, top_n: int = 5):
    a = load_artifacts()
    if wine_type not in a["wine_idx"]:
        raise HTTPException(status_code=404, detail=f"Unknown wine_type '{wine_type}'. See /wines.")

    scored = a["scored_by_wine"].get(wine_type, set())
    candidates = [f for f in a["food_items_sorted"] if f not in scored]
    preds = [(f, _predict_one(a, f, wine_type)) for f in candidates]
    preds.sort(key=lambda x: -x[1])
    top = preds[:top_n]

    return RecommendResponse(
        query_type="foods_for_wine",
        query_value=wine_type,
        recommendations=[{"food_item": f, "predicted_score": s} for f, s in top],
    )
