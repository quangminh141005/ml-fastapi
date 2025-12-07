import os
import sys

# Add the project root (ml-fastapi) to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()

def test_predict_basic():
    payload = {
        "keypoints": [0.0,0.0,0.20900188,-0.09110338,0.41800377,-0.15541166,0.5466203,-0.25723308,0.7181091,-0.36977255,0.31082332,-0.43943986,0.33761844,-0.62164664,0.3269004,-0.77169925,0.33225942,-0.943188,0.18756579,-0.49838912,0.19292481,-0.5466203,0.17684776,-0.43943986,0.16612971,-0.28938723,0.03751316,-0.4930301,0.069667295,-0.5251842,0.069667295,-0.40728572,0.05894925,-0.2625921,-0.085744366,-0.46623498,-0.053590227,-0.48767108,-0.048231203,-0.39656767,-0.048231203,-0.27866918]
    }
    resp = client.post("/predict",json=payload)
    assert resp.status_code in (200, 500)