# Xbox Series X multimodal validation and build handoff

The local integration is prepared against llama.cpp
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. Do not advance this pin without
re-auditing the mtmd API and build sources. No Series X performance or visual
quality result has been measured for this change. MSVC/UWP compilation, package
installation, and Open WebUI end-to-end testing remain required.

## Provision the first vision candidate

The matching files were re-verified in the ggml-org repository on 2026-09-27:

- [Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/blob/5037fcf163dd95d1e41d1974465f0898ed108ca2/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf)
  — SHA256 `d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12`.
- [mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/blob/5037fcf163dd95d1e41d1974465f0898ed108ca2/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf)
  — SHA256 `980c9b2f78c04e6cff93d277ada09e768394f112d75db3b4e9dea8a69f9fb904`.

Place both in `LocalState\models\qwen25-vl-3b\`. The language-model picker excludes
`mmproj` files; projector discovery requires exactly one matching-name candidate.
Keep matching weights together. A filename alone does not prove compatibility.
No weights are embedded in the application package.

For an experimental 8192-token context, merge this entry into the existing
`LocalState\manifest.json` `models` array, preserving other entries. Restart the
app after changing a loaded model's session policy. This is an experiment, not a
promotion into the shipped catalogue; console memory and quality gates still apply.

```json
{
  "models": [
    {
      "name": "qwen25-vl-3b",
      "display": "Qwen2.5-VL 3B (experimental)",
      "kind": "gguf",
      "n_ctx": 8192,
      "hf_base_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/5037fcf163dd95d1e41d1974465f0898ed108ca2",
      "files": [
        {
          "filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
          "remote": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
        },
        {
          "filename": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
          "remote": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
        }
      ]
    }
  ]
}
```

## Host verification

The standalone targets do not link WinRT or an inference runtime. When the pinned
submodule headers are available, they also compile the Session implementation and
run the bridge/decode contract against fake runtime functions. Those functions
are only linked into the test executable, never the Xbox application.

```powershell
cmake -S tests/host -B build/host-contracts -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-contracts --parallel 4
ctest --test-dir build/host-contracts --output-on-failure
./scripts/check-uwp-vision.ps1
```

Use a compiler appropriate for your machine; on Visual Studio omit `-G Ninja`,
use `-A x64`, and add `--config Release` / `-C Release` to build / ctest.
Assertions remain enabled in Release contract tests.

Local results: five CTest targets passed (SSE/KV, image parsing/order/presets,
model/projector selection, fake runtime prefill/decode contract, Python SSE
parser). The existing streaming doctest file also passed all nine cases and 434
assertions. The 48 mtmd library sources passed MinGW C++17 syntax checks with the
same miniaudio feature exclusions as the UWP project. These are not MSVC, UWP
linking, image-inference, or hardware proofs.

## Build wiring and commands on an authenticated machine

`uwp/xllama.vcxproj` defines `XLLAMA_HAS_MTMD=1` for every `unified` and `llamacpp`
build, includes mtmd headers, compiles `vision_mtmd.cpp`, and references the static
library. `uwp/ggml-uwp.vcxproj` compiles the full 48-source pinned mtmd library
alongside llama/ggml. There is no ffmpeg subprocess support. The explicit `ort`
variant uses the unsupported-vision stub.

`scripts/check-uwp-vision.ps1` checks the pin, source list, macro, include paths and
static reference before packaging. The app has a compile-time error if a
llama-enabled UWP build omits mtmd. Local builds default to `unified`.
`.github/actions/build-uwp-package/action.yml` runs the host contracts and existing
package flow. The UWP workflows recursively check out the pinned submodule.

The local push dry run failed because credentials were unavailable. The local
build stopped before signing at missing `vswhere.exe`/MSVC UWP tooling. No `.appx`
or `.msix` was produced. From this branch on an authenticated machine with push
permission to `origin`:

```bash
git switch codex/multimodal-uwp-ready
git submodule update --init --recursive
git push -u origin codex/multimodal-uwp-ready
gh workflow run build-uwp.yml --ref codex/multimodal-uwp-ready -f store_sku=false
gh run list --workflow build-uwp.yml --branch codex/multimodal-uwp-ready --limit 1
# Substitute the run ID printed above:
gh run watch RUN_ID --exit-status
gh run download RUN_ID --name xllama-appx --dir artifacts/xllama-appx
```

If `origin` is read-only, push the branch to your writable fork and dispatch the
same workflow there (`gh --repo OWNER/xllama ...`). A feature-branch push alone
does not trigger this workflow; open a PR or use the explicit dispatch above.
Investigate and fix actual compiler/linker errors before calling the package green.

On a Windows host with VS2022 C++ UWP, SDK 22621, Git Bash and NuGet:

```powershell
git submodule update --init --recursive
bash scripts/apply-uwp-patches.sh
./scripts/build-uwp.ps1 -Configuration Release -Platform x64 -Backend unified
```

Shipping runtime DLL patching follows the existing CI package action. Prefer its
`xllama-appx` artifact for console measurement. `git format-patch fe31af1..HEAD`
includes the previous Milestone B checkpoint and these completion fixes. Existing
local patched submodule files and untracked legacy verifier binaries are preserved
outside the commit; the gitlink remains unchanged.

## Run validation against the Series X

Keep xllama foregrounded, enable its LAN API, and stop competing clients during
measurements. Run the clients on the PC, using the Xbox address:

```bash
bash scripts/lan-stream-validate.sh --base-url http://XBOX_IP:11434 --model qwen25-vl-3b
python scripts/vision-validate.py --init-manifest cases.json
# Supply your nine images beside cases.json; add known OCR/value expectations.
python scripts/vision-validate.py --base-url http://XBOX_IP:11434 --model qwen25-vl-3b --manifest cases.json
```

The streaming client uses `curl -N`, timestamps SSE lines, validates role-first
ordering, multiple progressive chunks, UTF-8, finish reason, `[DONE]`, HTTP
chunking and CORS. It exercises JSON mode, resident turn 2, the same turn after a
reset, disconnect during generation, and subsequent recovery. It saves raw
responses, request IDs and timing evidence. Correlate those IDs with console
`api: request=... kv=reuse/reset`, result metrics and `stream-abort` logs to prove
backend KV reuse and cancellation. Timing or reconnect success alone is not proof.

The vision matrix runs Fast/Detailed and streaming/JSON for UI small text,
document OCR, chart, electronics schematic, normal scene, spatial relations,
table, Portuguese, and French. Review answers against the actual source images;
passing transport checks does not mean visual quality passed. Use `--case`,
`--preset`, and `--mode` to select individual cases.

Record these on **Series X**, with model hashes, package version, context and thread
settings, image dimensions, cold/warm state and repeated-run count:

| Measurement                                         | Evidence                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------- |
| Model/projector load and peak memory                | Console logs and Device Portal memory observations                          |
| Text JSON vs streamed decode tok/s and SSE overhead | Identical requests/settings; console token counts and decode times          |
| TTFT and turn-2 improvement                         | Client timestamps plus confirmed console KV reuse                           |
| Image preprocessing and vision prefill              | Console `vision` timing logs and visual token counts                        |
| Image-to-first-token and post-image decode tok/s    | Client TTFT plus console decode metrics; projector opens each image request |
| Fast vs Detailed                                    | Same source image, both actual mtmd token budgets                           |
| Visual quality                                      | Ground truth and manual review for all nine cases                           |

No substitute PC model benchmark establishes Xbox performance. Connect Open
WebUI to the Xbox `/v1` endpoint only after direct protocol validation, then
repeat text, image, streaming and multi-turn checks from the phone/notebook via
Open WebUI on the PC.
