import base64
import os
import sys
import tempfile

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.api.main import app


def _image(path: str, rgb: tuple[int, int, int]):
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    array[:, :, :] = rgb
    Image.fromarray(array).save(path)


def test_agent_routes_and_reports():
    directory = tempfile.gettempdir()
    first, second = os.path.join(directory, "agent_one.png"), os.path.join(directory, "agent_two.png")
    _image(first, (0, 180, 0))
    _image(second, (130, 130, 130))
    try:
        with TestClient(app) as client, open(first, "rb") as file:
            response = client.post("/api/v1/agent", files={"file_1": ("one.png", file, "image/png")}, data={"query": "Describe this scene", "include_report": "true"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["route"]["task"] == "caption"
        assert base64.b64decode(payload["report_pdf_b64"]).startswith(b"%PDF")
        with TestClient(app) as client, open(first, "rb") as one, open(second, "rb") as two:
            response = client.post("/api/v1/agent", files={"file_1": ("one.png", one, "image/png"), "file_2": ("two.png", two, "image/png")}, data={"query": "What changed between dates?"})
        assert response.status_code == 200, response.text
        assert response.json()["route"]["task"] == "change"
    finally:
        for path in (first, second):
            if os.path.exists(path):
                os.remove(path)
