# Portable single-node deployment

This guide deploys GLM-5.3-Flash EXL3 2.05 bpw with a DFlash2 drafter on one
NVIDIA GB10 system (DGX Spark or ASUS Ascent GX10). For several machines, run
one independent copy per machine and place an authenticated load balancer in
front of them. This is not a tensor-parallel multi-node recipe.

The repository contains no model weights, credentials, machine addresses,
user names, SSH configuration, or private infrastructure settings.

## Pinned profile

- Target: `turboderp/GLM-5.3-Flash-exl3`, revision
  `51058cd551c7e570d87bd32a4adee720edce2349`, quant `2.05bpw`
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- Published image: `ghcr.io/gitcommit90/glm-5.3-one-spark:general23`
- Served model ID: `GLM-5.3-Flash-EXL3`
- Context: 262,144 tokens
- DFlash2 depth: 5
- KV cache: FP8
- Concurrent sequences: 3
- Prefix caching: enabled with a fail-closed fix for vLLM #55600 / PR #55601

## Requirements

- One 128 GB NVIDIA GB10 machine
- Docker with NVIDIA Container Toolkit and working `--gpus all` support
- At least 220 GB of free disk during download/build
- Python 3 and the Hugging Face `hf` CLI

Verify Docker GPU access:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Install the Hugging Face CLI if required:

```bash
python3 -m pip install --user --upgrade huggingface_hub
```

## 1. Verify the source package

```bash
./scripts/verify-portable.sh
```

The verifier checks syntax, pinned revisions, launcher settings, the Mamba
patch's idempotence/fail-closed behavior, and common credential-file mistakes.

## 2. Download the pinned weights

Review `LICENSE`, `THIRD_PARTY_NOTICES.md`, and both model repository licenses.
DFlash2 is published under CC BY-NC-ND 4.0 for research/evaluation.

```bash
export ACCEPT_DFLASH2_NC_LICENSE=1
./download.sh
```

The default destinations are:

```text
$HOME/models/GLM-5.3-Flash-exl3-2.05bpw
$HOME/models/GLM-5.3-Flash-DFlash2
```

## 3. Start one node

The default binds only to loopback:

```bash
IMAGE=ghcr.io/gitcommit90/glm-5.3-one-spark:general23 \
ONE_SPARK_APC=1 \
ONE_SPARK_MAX_NUM_SEQS=3 \
./start.sh
```

Build the runtime image locally instead:

```bash
BUILD=1 ONE_SPARK_APC=1 ONE_SPARK_MAX_NUM_SEQS=3 ./start.sh
```

Use existing model directories:

```bash
MODEL_DIR=/absolute/path/to/GLM-5.3-Flash-exl3-2.05bpw \
DFLASH_DIR=/absolute/path/to/GLM-5.3-Flash-DFlash2 \
ONE_SPARK_APC=1 \
ONE_SPARK_MAX_NUM_SEQS=3 \
./start.sh
```

Follow startup and check the API:

```bash
docker logs -f glm53-one-spark
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18080/v1/models
```

The startup log must show:

```text
mamba_hybrid.py: mamba prefix seed patched (vLLM #55600)
enable_prefix_caching=True
```

If the source shape in a newer runtime image does not match the pinned build,
the patch aborts startup instead of guessing. Update and re-test the overlay;
do not bypass that failure.

## 4. Validate inference and prefix caching

```bash
curl --fail http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"GLM-5.3-Flash-EXL3",
    "messages":[{"role":"user","content":"Reply with OK only."}],
    "temperature":0,
    "max_tokens":16,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Run the included cold/warm cache test while the server is otherwise idle:

```bash
python3 tests/bench_prefix_cache.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model GLM-5.3-Flash-EXL3 \
  --runs 1 \
  --min-hit-ratio 0.80
```

Confirm that EngineCore did not restart:

```bash
docker inspect -f '{{.RestartCount}}' glm53-one-spark
```

Expected: HTTP 200, a substantial warm-request cache hit, and restart count 0.

## 5. Secure network access

Never expose an unauthenticated vLLM API to an untrusted network. Keep the
loopback default and use a TLS/authenticated gateway when possible. If direct
network binding is required, set a bearer key and restrict the port with a
firewall:

```bash
HOST=0.0.0.0 \
VLLM_API_KEY='replace-with-a-random-secret' \
ONE_SPARK_APC=1 \
ONE_SPARK_MAX_NUM_SEQS=3 \
./start.sh
```

Do not commit the real key to this repository.

## 6. Multiple independent nodes

Repeat sections 1–4 on every node, keeping the same served model ID. Configure
the gateway with one backend URL per node and health-check each backend.

Use rolling updates: drain one backend, upgrade and fully validate it, return it
to rotation, then repeat on the next node. Do not restart every backend at once.

## Operations and rollback

Stop the server without removing weights or compile caches:

```bash
./stop.sh
```

The container uses `restart: unless-stopped`, so Docker can recover it after a
process crash or host reboot. A deliberate `./stop.sh` remains stopped.

Emergency prefix-cache rollback:

```bash
ONE_SPARK_APC=0 ONE_SPARK_MAX_NUM_SEQS=3 ./start.sh
```

This disables prefix caching without changing the rest of the profile.
