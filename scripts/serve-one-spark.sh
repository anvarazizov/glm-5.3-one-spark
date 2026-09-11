#!/usr/bin/env bash
set -euo pipefail
# patch_glm_video_placeholders installs a .pth import hook into the live site-packages,
# so it must run at container start. The other overlay patches are applied at image build.
python3 /opt/glm53/patch_glm_video_placeholders.py
case "${ONE_SPARK_APC:-1}" in
  1)
    python3 /opt/glm53/patch_mamba_hybrid_seed.py
    PREFIX_CACHE_FLAG=--enable-prefix-caching
    ;;
  0)
    PREFIX_CACHE_FLAG=--no-enable-prefix-caching
    ;;
  *)
    echo 'ONE_SPARK_APC must be 0 or 1.' >&2
    exit 2
    ;;
esac
K="${ONE_SPARK_K:-5}"  # DFlash2 draft depth; 5 = best prose/code, 8 = best structured (see README K sweep)
SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":'"$K"',"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":1}'
exec vllm serve /model \
  --served-model-name GLM-5.3-Flash-EXL3 \
  --host "${ONE_SPARK_HOST:-127.0.0.1}" --port "${ONE_SPARK_PORT:-18080}" \
  --tensor-parallel-size 1 \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  "$PREFIX_CACHE_FLAG" --no-enable-flashinfer-autotune \
  --quantization exl3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs "${ONE_SPARK_MAX_NUM_SEQS:-3}" --max-num-batched-tokens 7168 \
  --kv-cache-dtype fp8 \
  --speculative-config "$SPEC" \
  --chat-template /opt/glm53/chat_template.jinja \
  --limit-mm-per-prompt '{"image":2,"video":0}' --skip-mm-profiling \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32
