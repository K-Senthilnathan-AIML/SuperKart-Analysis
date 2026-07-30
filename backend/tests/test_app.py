import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import app as backend_app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = backend_app.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_predict_endpoint(self):
        payload = {
            "Product_Weight": 12.5,
            "Product_Sugar_Content": "Low Sugar",
            "Product_Allocated_Area": 0.03,
            "Product_MRP": 120.0,
            "Store_Size": "Small",
            "Store_Location_City_Type": "Tier 1",
            "Store_Type": "Supermarket Type1",
            "Product_Id_char": "FD",
            "Store_Age_Years": 10,
            "Product_Type_Category": "Perishables",
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.get_json())


if __name__ == "__main__":
    unittest.main()
