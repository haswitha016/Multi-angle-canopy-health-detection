import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as canopy_app


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.upload_dir = root / "uploads"
        self.report_dir = root / "reports"
        canopy_app.app.config.update(TESTING=True)
        self.client = canopy_app.app.test_client()
        self.upload_patch = patch.object(canopy_app, "UPLOAD_DIR", self.upload_dir)
        self.report_patch = patch.object(canopy_app, "REPORT_DIR", self.report_dir)
        self.model_patch = patch.object(canopy_app, "MODEL_PATH", root / "missing.pth")
        self.upload_patch.start()
        self.report_patch.start()
        self.model_patch.start()

    def tearDown(self):
        self.upload_patch.stop()
        self.report_patch.stop()
        self.model_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def make_image(color=(30, 150, 45)):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 240), color).save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer

    def test_home_page_renders_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DEMO MODE", response.data)

    def test_demo_upload_returns_fusion_and_downloadable_report(self):
        response = self.client.post(
            "/api/analyze",
            data={
                "mode": "demo",
                "quality_weighted": "false",
                "view_1": "Front",
                "images": (self.make_image(), "front.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        report = response.get_json()
        self.assertEqual(report["mode"], "demo")
        self.assertEqual(report["images"][0]["view"], "Front")
        self.assertEqual(
            set(report["fusion"]),
            {"majority_vote", "average_probability", "confidence_weighted"},
        )
        download = self.client.get(report["report_url"])
        self.assertEqual(download.status_code, 200)
        saved_report = json.loads(download.get_data(as_text=True))
        self.assertEqual(saved_report["report_id"], report["report_id"])
        download.close()

    def test_trained_mode_without_checkpoint_returns_actionable_error(self):
        response = self.client.post(
            "/api/analyze",
            data={"mode": "trained_model", "images": (self.make_image(), "view.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Train the model first", response.get_json()["error"])

    def test_rejects_corrupt_image_upload(self):
        response = self.client.post(
            "/api/analyze",
            data={"mode": "demo", "images": (io.BytesIO(b"not an image"), "broken.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not decode", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()