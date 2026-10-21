import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from maker.__main__ import main, parser, write_manifest
from maker.models import MakerError, ensure_model, verify


class ArgumentsTest(unittest.TestCase):
    def test_requested_command(self):
        args = parser().parse_args(["--hospital", "smc", "--grade", "3", "--num", "100"])
        self.assertEqual((args.hospital, args.grade, args.num, args.seed, args.impl), ("smc", 3, 100, 0, "ref"))

    def test_invalid_conditions(self):
        for options in (["--grade", "6"], ["--grade", "-1"], ["--num", "0"], ["--num", "10001"], ["--seed", "-1"], ["--trunc", "nan"], ["--trunc", "inf"], ["--hospital", "unknown"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(options)

    @patch("maker.__main__.registry", return_value={"models": {}})
    def test_seed_range_overflow(self, _):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--hospital", "smc", "--grade", "0", "--seed", "4294967295", "--num", "2"])

    @patch("maker.__main__.registry", return_value={"models": {}})
    def test_existing_output_is_preserved(self, _):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("keep")
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(["--hospital", "smc", "--grade", "0", "--outdir", temp])
            self.assertEqual(result, 1)
            self.assertEqual(marker.read_text(), "keep")


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.content = b"test model bytes"
        self.spec = {"filename": "test.safetensors", "bytes": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest(), "url": "https://example.org/test"}

    def response(self, content=None):
        response = io.BytesIO(self.content if content is None else content)
        response.url = self.spec["url"]
        return response

    def test_valid_offline_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / self.spec["filename"]
            path.write_bytes(self.content)
            with patch("urllib.request.urlopen") as request:
                self.assertEqual(ensure_model(self.spec, Path(temp), offline=True), path)
                request.assert_not_called()

    def test_same_length_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / self.spec["filename"]
            path.write_bytes(b"x" * len(self.content))
            with self.assertRaisesRegex(MakerError, "verification failed"):
                verify(path, self.spec)

    def test_missing_offline_never_downloads(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(MakerError, "Offline"):
                ensure_model(self.spec, Path(temp), offline=True)
            request.assert_not_called()

    def test_pending_release_never_downloads(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(MakerError, "not yet released"):
                ensure_model(dict(self.spec, url=None), Path(temp))
            request.assert_not_called()

    def test_successful_atomic_download(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=self.response()):
            path = ensure_model(self.spec, Path(temp))
            self.assertEqual(path.read_bytes(), self.content)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_corrupt_download_leaves_no_cache_entry(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=self.response(b"corrupt")):
            with self.assertRaises(MakerError):
                ensure_model(self.spec, Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_oversize_download_leaves_no_cache_entry(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=self.response(self.content * 2)):
            with self.assertRaisesRegex(MakerError, "exceeds"):
                ensure_model(self.spec, Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_insecure_redirect_rejected(self):
        response = self.response()
        response.url = "http://example.org/test"
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(MakerError, "insecure"):
                ensure_model(self.spec, Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])


class ManifestTest(unittest.TestCase):
    def test_running_and_completed_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            write_manifest(path, {"status": "running", "images": []})
            write_manifest(path, {"status": "complete", "images": [{"seed": 3}]})
            self.assertEqual(json.loads(path.read_text())["images"], [{"seed": 3}])
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
