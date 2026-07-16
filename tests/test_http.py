import json
import os
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "server")
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from app import Handler


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def post(self, path, payload):
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.headers, response.read().decode()

    def test_index_and_skills(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            html = response.read().decode()
        self.assertIn('class="cols"', html)
        self.assertIn('id="thinkPanel"', html)
        self.assertIn("const STRAT_LABEL", html)
        with urllib.request.urlopen(self.base + "/api/skills", timeout=5) as response:
            data = json.load(response)
        ids = {s["id"] for s in data["skills"]}
        self.assertGreaterEqual(len(data["skills"]), 5)
        self.assertNotIn("user_material_understanding", ids)
        self.assertNotIn("viral_reference_understanding", ids)

    def test_trends_and_two_sse_stages(self):
        _, trend_body = self.post("/api/trends", {"industry_id": "ecom"})
        self.assertGreaterEqual(len(json.loads(trend_body)["trends"]), 8)
        headers, body = self.post("/api/analyze", {"skill_id": "ecom_product", "materials": [], "use_cache": False})
        self.assertTrue(headers.get_content_type().startswith("text/event-stream"))
        self.assertIn('"type": "reasoning"', body)
        self.assertIn('"type": "analysis"', body)
        analysis_line = next(line[6:] for line in body.splitlines() if '"type": "analysis"' in line)
        analysis = json.loads(analysis_line)["result"]
        _, final_body = self.post("/api/replicate", {"industry_id": "ecom", "template": analysis["template"], "material_strategy": "faithful"})
        self.assertIn('"type": "reasoning"', final_body)
        self.assertIn('"type": "step"', final_body)
        self.assertIn('"type": "final"', final_body)
        self.assertIn("data: [DONE]", final_body)

    def test_multipart_upload(self):
        boundary = "----testboundary"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"demo.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n--{boundary}--\r\n").encode()
        req = urllib.request.Request(self.base + "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.load(response)
        self.assertTrue(data["video_uri"].startswith("uploads/"))
        self.assertEqual(data["size"], 5)
        os.remove(os.path.join(ROOT, data["video_uri"]))


if __name__ == "__main__":
    unittest.main()
