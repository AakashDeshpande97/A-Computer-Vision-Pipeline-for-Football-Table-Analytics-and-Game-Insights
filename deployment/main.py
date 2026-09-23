
# CI/CD pipeline test - automated deploy via GitHub Actions
"""
FastAPI wrapper around foosball_core.py.

Accepts a video upload, runs the full pipeline (calibration correction
-> ArUco homography -> ball/player tracking -> hit detection ->
sequence building -> TCN model), returns the winner prediction.

Calibration files and model weights are baked into the Docker image
at build time (see Dockerfile) since this service is calibrated for
one specific physical camera/table setup.
"""
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from foosball_core import load_calibration, load_model, predict_winner

app = FastAPI(
    title="Foosball Winner Prediction API",
    description="Upload a match video from the calibrated camera setup; get back a winner prediction.",
)

CALIB_DIR = Path(__file__).parent / "calibration_artifacts"
MODEL_PATH = Path(__file__).parent / "model_weights" / "TCN_best.pt"

# Loaded once at startup, reused across requests.
_state = {}


@app.on_event("startup")
def startup_event():
    K_new, map1, map2 = load_calibration(str(CALIB_DIR))
    model = load_model(str(MODEL_PATH))
    _state["K_new"] = K_new
    _state["map1"] = map1
    _state["map2"] = map2
    _state["model"] = model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if not video.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Upload a .mp4, .mov, or .avi file")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        start = time.time()
        result = predict_winner(
            str(tmp_path), _state["K_new"], _state["map1"], _state["map2"], _state["model"]
        )
        elapsed = time.time() - start
        result["processing_time_s"] = round(elapsed, 1)
    except ValueError as e:
        # Expected failure modes: ArUco markers not detected, ball not tracked, etc.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return result
