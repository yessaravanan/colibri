# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

colibri runs very large open mixture-of-experts models on ordinary machines. The hot weights (attention, shared experts, router) stay in RAM and the routed experts are streamed from disk on demand, in dependency-free C with OpenMP, no GPU required. Optional backends add CUDA, Metal, Vulkan (any GPU) and XDNA. Each model family has its own engine, usually one big `.c` file, plus a Python launcher and OpenAI-compatible gateway on top.

## Layout

Almost everything lives in `c/`. The root `Makefile` just forwards `all glm deepseek-v4 portable test check cuda-test clean install uninstall` to `make -C c`. Other top-level dirs: `web/` (React/Vite dashboard), `ui/` (single-page chat UI, stdlib-only Python server, no build step), `desktop/` (Tauri shell), `docker/`, `docs/` (per-engine docs, `ENVIRONMENT.md`, `SETTINGS.md`), `site/`.

## Commands

Run from the repo root or `c/`; both work.

```sh
make check                 # clean + portable CPU build + C tests + Python tests (what CONTRIBUTING asks for; no model, no CUDA)
make test-c                # C unit tests only (tools/run_tests.py over tests/test_*.c binaries)
make test-python           # python3 -m unittest discover -s tests -p 'test_*.py'
make test-asan             # C suite under ASan+UBSan (leak detection intentionally off)
make cuda-test CUDA_ARCH=native   # CUDA host only
make -C c colibri          # build one engine (also: glm53 inkling kimi_k3 olmoe qwen36 qwen38 qwenimage deepseek_v41 mimo laya gliner_decide)
make -C c deepseek-v4      # DeepSeek V4 builds via Makefile.deepseek-v4, separate from the main engine list
make -C c qwen38-tiny-check   # per-engine "tiny" fixture checks (also qwen36-tiny-*, kimi-k3-tiny-check, deepseek-v4-tiny-check)
```

Run a single Python test with `cd c && python3 -m unittest tests.test_foo` (or `tests.test_foo.Class.test_name`). Run a single C test by building its binary, e.g. `make tests/test_affine_quant` and executing it. Add compiler flags with `EXTRA_CFLAGS=...`, never by overriding `CFLAGS=`, which drops the platform's OpenMP/`-march` flags. The build must be 0 warnings.

The correctness gate for the GLM engine is the oracle, run from `c/` after generating the `glm_tiny` fixture with `python3 c/tools/make_glm_oracle.py` (needs torch):

```sh
SNAP=./glm_tiny TF=1 COLI_TEMP=0 ORACLE_STRICT=1 ORACLE_TF_MAX_MISMATCHES=2 ./colibri 64 16 16   # teacher-forced
SNAP=./glm_tiny COLI_TEMP=0 ORACLE_STRICT=1 ./colibri 64 16 16                                   # greedy, must be token-exact
```

Web dashboard: `cd web && npm install && npm run dev | npm test | npm run build`.

## Architecture

`c/coli` is the Python launcher and the user entry point (`setup`, `chat`, `serve`, `run`, `plan`, `doctor`, `bench`, `convert`, `ui`...). It picks the engine binary for a model directory via `family_registry.py`, sizes RAM/VRAM budgets with `resource_plan.py`, and spawns the engine as a subprocess. `openai_server.py` is the persistent gateway (`coli serve`): it holds one engine process and speaks the OpenAI/Anthropic-style HTTP APIs, with per-family chat-template and tool-call handling (`v4_dsml.py`, `v41_dsml.py`). `setup_*.py` and `autotune.py` handle hardware detection, model catalog, download and tuning. `edge_runtime.c` and `segment_runtime.c` are shared runtimes; `expert_store_registry.c` is the shared expert-storage abstraction.

Engines are per-family, not generic: `colibri.c` (GLM-5.2, also Mistral-Large-4), `glm53.c`, `inkling.c`, `kimi_k3.c`, `deepseek_v4.c` / `deepseek_v41.c`, `mimo.c`, `qwen36.c` (also serves Qwen3-Coder, dense Qwen3.8, Flint, Muse Spark, LFM3-VL and the Clef decision model), `qwen38.c`, `olmoe.c`, `qwenimage.c`, plus `laya.c` and `gliner_decide.c`. Shared pieces are header-only or small units: `st.h` (safetensors indexing/mmap), `json.h`, `compat.h` (Windows/POSIX shims), `qpack.*` and `affine_quant.h` (quant formats), `rans.h`, GGUF readers, and the tier/cache code (`cu_tier.*`, `vk_tier.c`, `qwen36_tier.*`) that places hot experts in VRAM over a RAM cache over disk. GPU backends are `backend_cuda*.cu`, `backend_metal.mm`, `backend_vulkan.c`, `backend_xdna.c`, selected through `backend_loader*.c`.

Platform detection in `c/Makefile` keys off `$(CC) -dumpmachine`, not `uname`, so cross-compiles and native Windows (MinGW UCRT64 only; MSYS/Cygwin gcc is rejected) work. LTO is opt-in because FMA contraction changes could break the token-exact oracles, so keep the default build's floating-point behavior stable.

Tests in `c/tests/` come in three kinds: `test_*.c` unit binaries, `test_*.py` unittest modules, and `*_harness.py` / `*_check` Makefile targets that drive an engine against a generated tiny fixture. Fixtures are generated, not committed (see `*-tiny-generate` targets).

## Contributing rules worth knowing

Open PRs against `dev`, not `main`; `main` must always pass the oracle. Keep the default CPU path dependency-free. Env-var knobs are documented in `docs/ENVIRONMENT.md` and CLI flags in `docs/SETTINGS.md`; both are generated from `getenv(...)` and `add_argument(...)` call sites, so follow `docs/MAINTAINING-DOCS.md` when adding knobs. Performance PRs need an experiment manifest validated with `python3 c/experiment_manifest.py <result.json>`.
