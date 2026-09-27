"""Wire parser contract tests; no inference backend or network required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lan_validate import CASES, Events, ValidationError, validate_headers


def event(delta, finish=None):
    return json.dumps({"id": "test", "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


class WireContracts(unittest.TestCase):
    def test_progressive_unicode_and_done(self):
        state = Events()
        for payload, at in [(event({"role": "assistant"}), 0),
                            (event({"content": "ação "}), 0.1),
                            (event({"content": "français 🙂"}), 0.2),
                            (event({}, "stop"), 0.3), ("[DONE]", 0.4)]:
            state.accept(payload, at)
        result = state.result()
        self.assertEqual(result["text"], "ação français 🙂")
        self.assertEqual(result["content_chunks"], 2)
        self.assertGreater(result["delivery_span_seconds"], 0)
        with self.assertRaises(ValidationError):
            state.accept("[DONE]", 0.5)

    def test_duplicate_role_and_premature_done_fail(self):
        state = Events()
        state.accept(event({"role": "assistant"}), 0)
        with self.assertRaises(ValidationError):
            state.accept(event({"role": "assistant"}), 0.1)
        with self.assertRaises(ValidationError):
            state.accept("[DONE]", 0.2)

    def test_missing_role_and_invalid_unicode_fail(self):
        with self.assertRaises(ValidationError):
            Events().accept(event({"content": "hello"}), 0)
        state = Events()
        state.accept(event({"role": "assistant"}), 0)
        with self.assertRaises(UnicodeError):
            state.accept(event({"content": "\ud800"}), 0.1)

    def test_header_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "headers.txt"
            good = ("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Transfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\n\r\n")
            path.write_bytes(good.encode("ascii"))
            validate_headers(path, True)
            path.write_bytes(good.replace("Transfer-Encoding: chunked", "Content-Length: 42").encode("ascii"))
            with self.assertRaises(ValidationError):
                validate_headers(path, True)

    def test_nine_required_vision_cases(self):
        self.assertEqual(len(CASES), 9)
        self.assertEqual(len({case[0] for case in CASES}), 9)


if __name__ == "__main__":
    unittest.main()
