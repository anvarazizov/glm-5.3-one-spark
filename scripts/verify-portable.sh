#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash -n start.sh stop.sh download.sh scripts/serve-one-spark.sh
VERIFY_PYCACHE="${TMPDIR:-/tmp}/glm53-one-spark-portable-pycache"
PYTHONPYCACHEPREFIX="$VERIFY_PYCACHE" python3 -m py_compile overlay/*.py tests/*.py
PYTHONPYCACHEPREFIX="$VERIFY_PYCACHE" python3 tests/test_mamba_hybrid_seed.py

grep -Fq "TARGET_REV='51058cd551c7e570d87bd32a4adee720edce2349'" download.sh
grep -Fq "DRAFT_REV='bf582e4eacc1810f76656d1811693ff6c6737d2a'" download.sh
grep -Fq 'ONE_SPARK_MAX_NUM_SEQS:-3' scripts/serve-one-spark.sh
grep -Fq 'ONE_SPARK_APC:-1' scripts/serve-one-spark.sh
grep -Fq -- '--served-model-name GLM-5.3-Flash-EXL3' scripts/serve-one-spark.sh

for forbidden in .env id_rsa id_ed25519 authorized_keys known_hosts; do
  if find . -type f -name "$forbidden" -print -quit | grep -q .; then
    echo "Forbidden credential/access file found: $forbidden" >&2
    exit 1
  fi
done

if grep -RIEq --exclude='verify-portable.sh' \
  '(BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}|hf_[A-Za-z0-9]{20,}|eyJ[a-zA-Z0-9_-]+\.eyJ)' .; then
  echo 'Possible embedded credential found.' >&2
  exit 1
fi

echo 'Portable deployment verification passed.'
