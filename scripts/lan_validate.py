#!/usr/bin/env python3
"""LAN-only validation clients. curl -N supplies bytes; Python timestamps receipt.

Artifacts distinguish wire observations from backend evidence. Timing does not
prove KV reuse, and client disconnect/recovery does not prove an abort flag.
Run without other clients changing the Xbox resident conversation.
"""
import argparse
import base64
import datetime
import json
import mimetypes
import pathlib
import subprocess
import sys
import time


class ValidationError(RuntimeError):
    pass


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Events:
    """Strict OpenAI SSE state machine; timestamps are captured before parsing."""
    def __init__(self):
        self.events = 0
        self.text = ""
        self.content_times = []
        self.finish = None
        self.done = False
        self.request_id = None

    def accept(self, payload, elapsed):
        if self.done:
            raise ValidationError("data after [DONE]")
        if payload == "[DONE]":
            if self.finish is None:
                raise ValidationError("[DONE] before finish_reason")
            self.done = True
            return
        obj = json.loads(payload)
        if obj.get("object") != "chat.completion.chunk":
            raise ValidationError("expected chat.completion.chunk: " + payload[:200])
        if not self.events:
            self.request_id = obj.get("id")
        if not self.request_id or obj.get("id") != self.request_id:
            raise ValidationError("missing or changing completion id")
        choice = obj["choices"][0]
        if choice.get("index") != 0:
            raise ValidationError("unexpected choice index")
        delta = choice["delta"]
        if self.events == 0 and (delta.get("role") != "assistant" or delta.get("content")):
            raise ValidationError("first event must be assistant role before content")
        if self.events and "role" in delta:
            raise ValidationError("repeated role event (possible duplicate socket write)")
        if self.finish is not None:
            raise ValidationError("JSON event after finish_reason")
        content = delta.get("content", "")
        if not isinstance(content, str):
            raise ValidationError("content delta must be text")
        # Also reject escaped lone surrogates and replacement artifacts.
        content.encode("utf-8", errors="strict")
        if "\ufffd" in content:
            raise ValidationError("replacement character in model output; inspect UTF-8 path")
        if content:
            self.text += content
            self.content_times.append(elapsed)
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish not in ("stop", "length", "content_filter", "tool_calls"):
                raise ValidationError("unknown finish_reason")
            self.finish = finish
        self.events += 1

    def result(self, require_done=True):
        if not self.events or not self.content_times:
            raise ValidationError("no assistant content received")
        if require_done and (not self.done or self.finish is None):
            raise ValidationError("incomplete stream (finish_reason and [DONE] required)")
        return {
            "id": self.request_id, "text": self.text, "events": self.events,
            "content_chunks": len(self.content_times), "finish_reason": self.finish,
            "done": self.done, "ttft_seconds": self.content_times[0],
            "delivery_span_seconds": self.content_times[-1] - self.content_times[0],
        }


def validate_headers(path, streaming):
    headers = path.read_text(encoding="iso-8859-1").lower().split("\n\n")
    blocks = [block for block in headers if block.startswith("http/")]
    if not blocks or " 200 " not in blocks[-1].splitlines()[0]:
        raise ValidationError("non-200 HTTP response: " + (blocks[-1][:150] if blocks else "none"))
    fields = dict(line.split(":", 1) for line in blocks[-1].splitlines()[1:] if ":" in line)
    if streaming:
        if "text/event-stream" not in fields.get("content-type", ""):
            raise ValidationError("missing text/event-stream content type")
        if "chunked" not in fields.get("transfer-encoding", "") or "content-length" in fields:
            raise ValidationError("stream must use HTTP chunked without Content-Length")
    if "access-control-allow-origin" not in fields:
        raise ValidationError("missing CORS response header")


def request(args, label, payload, disconnect_chunks=0):
    folder = args.output / label
    folder.mkdir(parents=True, exist_ok=False)
    body_path, headers_path = folder / "request.json", folder / "headers.txt"
    save_json(body_path, payload)
    command = ["curl", "-N", "--http1.1", "--silent", "--show-error",
               "--connect-timeout", "10", "--max-time", str(args.timeout),
               "--dump-header", str(headers_path), "--header", "Content-Type: application/json",
               "--data-binary", "@" + str(body_path), args.base_url.rstrip("/") + "/v1/chat/completions"]
    start = time.monotonic()
    interrupted = False
    with (folder / "curl.stderr").open("wb") as errors, (folder / "body.raw").open("wb") as raw:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            if payload["stream"]:
                state = Events()
                data_lines = []
                with (folder / "sse-timestamps.jsonl").open("w", encoding="utf-8") as trace:
                    for line in iter(proc.stdout.readline, b""):
                        elapsed = time.monotonic() - start
                        raw.write(line)
                        text = line.decode("utf-8", errors="strict").rstrip("\r\n")
                        trace.write(json.dumps({"elapsed_seconds": elapsed, "line": text}, ensure_ascii=False) + "\n")
                        trace.flush()
                        print(f"[{label} +{elapsed:.6f}s] {text}", flush=True)
                        if text.startswith("data:"):
                            data_lines.append(text[5:].lstrip(" "))
                        elif not text and data_lines:
                            state.accept("\n".join(data_lines), elapsed)
                            data_lines.clear()
                            if disconnect_chunks and len(state.content_times) >= disconnect_chunks:
                                interrupted = True
                                proc.terminate()  # close live curl/socket before finish and DONE
                                break
                    if data_lines and not interrupted:
                        raise ValidationError("unterminated SSE event")
                result = state.result(require_done=not interrupted)
            else:
                data = proc.stdout.read()
                raw.write(data)
                obj = json.loads(data.decode("utf-8", errors="strict"))
                if obj.get("object") != "chat.completion":
                    raise ValidationError("expected chat.completion: " + str(obj)[:250])
                choice = obj["choices"][0]
                if choice["message"].get("role") != "assistant" or not choice["message"].get("content"):
                    raise ValidationError("missing assistant response")
                if not choice.get("finish_reason"):
                    raise ValidationError("missing finish_reason")
                result = {"id": obj.get("id"), "text": choice["message"]["content"],
                          "finish_reason": choice["finish_reason"], "usage": obj.get("usage")}
            exit_code = proc.wait(timeout=10)
            if exit_code and not interrupted:
                raise ValidationError(f"curl exit {exit_code}; inspect {folder / 'curl.stderr'}")
            validate_headers(headers_path, payload["stream"])
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
    result.update(elapsed_seconds=time.monotonic() - start, client_disconnected=interrupted)
    save_json(folder / "result.json", result)
    return result


def make_payload(args, messages, stream=True):
    return {"model": args.model, "messages": messages, "stream": stream,
            "temperature": 0, "seed": 42, "max_tokens": args.max_tokens}


def run_stream(args):
    system = {"role": "system", "content": "Follow instructions precisely. Preserve accents and Unicode."}
    user = {"role": "user", "content":
            "Start with exactly: café, ação, français, 🙂. Then write 80 numbered short facts about the ocean."}
    first_payload = make_payload(args, [system, user])
    first = request(args, "turn1-stream", first_payload)
    if first["content_chunks"] < 2 or first["delivery_span_seconds"] < args.min_delivery_span:
        raise ValidationError("progressive delivery unproven; need multiple content chunks spread over time")
    if not any(ord(c) > 127 for c in first["text"]):
        raise ValidationError("model emitted no non-ASCII text, so UTF-8 demonstration is inconclusive")
    messages = [system, user, {"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "Summarize your ocean facts in three sentences."}]
    turn2_payload = make_payload(args, messages)
    turn2 = request(args, "turn2-resident", turn2_payload)
    # An unrelated history forces reset. Then replay the EXACT turn-2 request to
    # compare full prefill with continuation, without substituting a host model.
    request(args, "reset-between", make_payload(args, [{"role": "user", "content": "Say ready."}], False))
    cold = request(args, "turn2-full-prefill", turn2_payload)
    nonstream = request(args, "text-nonstream", make_payload(args, [system, user], False))
    abort_payload = make_payload(args, [{"role": "user", "content": "Count from 1 to 2000, spelling out every number."}])
    abort_payload["max_tokens"] = args.disconnect_max_tokens
    aborted = request(args, "disconnect", abort_payload, disconnect_chunks=2)
    if not aborted["client_disconnected"] or aborted["done"]:
        raise ValidationError("request finished before disconnect; abort exercise inconclusive")
    recovery_start = time.monotonic()
    recovery = None
    attempt = 0
    while time.monotonic() - recovery_start < args.recovery_timeout:
        attempt += 1
        try:
            recovery = request(args, f"recovery-{attempt}", make_payload(args, [{"role": "user", "content": "Say recovered."}], False))
            break
        except ValidationError:
            # Only a busy response is expected during cancellation; preserve all evidence.
            headers = (args.output / f"recovery-{attempt}" / "headers.txt").read_text()
            if "503" not in headers:
                raise
            time.sleep(0.5)
    if recovery is None:
        raise ValidationError("endpoint did not recover before recovery timeout")
    summary = {
        "wire_checks": "PASS: role-first, multiple timed chunks, finish, DONE, UTF-8, chunked, CORS",
        "turn1": first, "turn2_resident": turn2, "turn2_full_prefill": cold,
        "nonstream": nonstream, "disconnect": aborted,
        "recovery_seconds": time.monotonic() - recovery_start,
        "backend_kv_reuse": "UNVERIFIED: correlate response ids with Xbox api kv=reuse/reset logs",
        "backend_disconnect_abort": "UNVERIFIED: correlate interrupted response id with Xbox stream-abort log",
        "timing_note": "Client receive times include LAN and scheduling; no inference tok/s or memory claim.",
    }
    save_json(args.output / "summary.json", summary)
    print(f"Wire validation passed. Xbox backend abort/KV evidence still required. Results: {args.output}")


CASES = [
    ("ui-small-text", "ui-small-text.png", "Transcribe all small UI labels and describe the selected control.", "Small text and selected-state accuracy"),
    ("document-ocr", "document-ocr.png", "Transcribe this document faithfully, preserving line order and punctuation.", "Character and line-order accuracy"),
    ("chart", "chart.png", "Read the axes, legend and values. Explain the main trend without inventing numbers.", "Labels, numeric values, trend"),
    ("schematic", "schematic.png", "Identify components, labels and connections in this electronics schematic.", "Component labels and actual connectivity"),
    ("scene", "scene.jpg", "Describe the visible scene, objects and actions.", "Visible object accuracy and hallucinations"),
    ("spatial", "spatial.png", "Describe which objects are left, right, above, below, in front of and behind others.", "Spatial relation correctness"),
    ("table", "table.png", "Transcribe this table as Markdown with exact headers and cell values.", "Row/column association and exact values"),
    ("portuguese", "portuguese.png", "Transcreva o texto em português exatamente, preservando acentos e pontuação.", "Portuguese diacritics and exact wording"),
    ("french", "french.png", "Transcrivez exactement le texte français, en conservant les accents et la ponctuation.", "French diacritics and exact wording"),
]


def run_vision(args):
    if args.init_manifest:
        save_json(args.init_manifest, [{"id": key, "image": image, "prompt": prompt,
                                       "review": review, "expected_substrings": []}
                                      for key, image, prompt, review in CASES])
        print(f"Created {args.init_manifest}; put your nine images beside it and add expected text when useful.")
        return
    if not args.manifest:
        raise ValidationError("--manifest is required; create one with --init-manifest cases.json")
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [case for case in cases if not args.case or case["id"] in args.case]
    if not selected:
        raise ValidationError("no matching cases")
    results = []
    for case in selected:
        image = args.manifest.parent / case["image"]
        if not 0 < image.stat().st_size <= 16 * 1024 * 1024:
            raise ValidationError(f"empty/oversized image: {image}")
        mime = mimetypes.guess_type(image.name)[0]
        if mime not in ("image/png", "image/jpeg", "image/gif"):
            raise ValidationError(f"unsupported image extension: {image}")
        url = "data:" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode("ascii")
        for preset in args.preset or ("fast", "detailed"):
            for mode in args.mode or ("stream", "nonstream"):
                parts = [{"type": "text", "text": case["prompt"]}, {"type": "image_url", "image_url": {"url": url}}]
                payload = make_payload(args, [{"role": "user", "content": parts}], mode == "stream")
                payload["vision_preset"] = preset
                label = f"{case['id']}-{preset}-{mode}"
                if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in label):
                    raise ValidationError("case ids must contain only letters, digits, hyphens or underscores")
                result = request(args, label, payload)
                result.update(case=case["id"], preset=preset, mode=mode, quality_review="PENDING: " + case.get("review", "review against source image"))
                result["missing_expected_substrings"] = [s for s in case.get("expected_substrings", []) if s not in result["text"]]
                results.append(result)
                save_json(args.output / "vision-summary.json", results)
    print(f"Completed {len(results)} Xbox requests. Review answers against source images: {args.output}")
    if any(r["missing_expected_substrings"] for r in results):
        raise ValidationError("one or more expected text checks failed; inspect vision-summary.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stream", "vision"))
    parser.add_argument("--base-url", help="Xbox LAN API URL, e.g. http://192.168.1.50:11434")
    parser.add_argument("--model", help="Provisioned Xbox model id/path")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("lan-results-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--timeout", type=float, default=900, help="curl timeout per request (seconds)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-delivery-span", type=float, default=0.01, help="Minimum first-to-last content receive time to demonstrate progressive delivery")
    parser.add_argument("--recovery-timeout", type=float, default=30)
    parser.add_argument("--disconnect-max-tokens", type=int, default=1024, help="Must fit Xbox n_ctx with disconnect prompt")
    parser.add_argument("--manifest", type=pathlib.Path, help="Vision case manifest; images are relative to it")
    parser.add_argument("--init-manifest", type=pathlib.Path, help="Write all nine vision case definitions without sending requests")
    parser.add_argument("--case", action="append", help="Vision case id (repeatable); default all nine")
    parser.add_argument("--preset", action="append", choices=("fast", "detailed"))
    parser.add_argument("--mode", action="append", choices=("stream", "nonstream"))
    args = parser.parse_args(argv)
    if not args.init_manifest and (not args.base_url or not args.model):
        parser.error("--base-url and --model are required for Xbox requests")
    try:
        if not args.init_manifest:
            args.output = args.output.resolve()
            args.output.mkdir(parents=True, exist_ok=False)
        if args.command == "stream":
            run_stream(args)
        else:
            run_vision(args)
    except (ValidationError, OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
