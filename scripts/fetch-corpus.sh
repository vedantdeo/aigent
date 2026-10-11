#!/usr/bin/env bash
# Download the Project 1 corpus and the fine-tune's training reports into corpus/, verifying each
# digest against its manifest.
# The PDFs are not in git: too large, and not ours to redistribute.
set -euo pipefail

cd "$(dirname "$0")/.."
MANIFESTS="evals/reference/corpus-manifest.json evals/reference/training-manifest.json"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"

mkdir -p corpus corpus/training
failed=0

while IFS=$'\t' read -r dir doc_id url want; do
  path="${dir}/${doc_id}.pdf"
  if [ -f "$path" ] && [ "$(shasum -a 256 "$path" | cut -d' ' -f1)" = "$want" ]; then
    echo "ok       ${doc_id} (already downloaded)"
    continue
  fi
  echo "fetching ${doc_id} ..."
  if ! curl -fsSL --max-time 600 -A "$UA" -e "$(echo "$url" | cut -d/ -f1-3)/" -o "$path" "$url"; then
    echo "FAILED   ${doc_id}: could not download. Annual report URLs move; find the current one" >&2
    echo "         and update its entry in ${MANIFESTS// / or }." >&2
    rm -f "$path"
    failed=1
    continue
  fi
  got=$(shasum -a 256 "$path" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "FAILED   ${doc_id}: digest ${got} does not match the manifest. The company has" >&2
    echo "         republished the report; the retrieval numbers were measured on the old one." >&2
    failed=1
    continue
  fi
  echo "ok       ${doc_id}"
done < <(python3 -c "
import json, sys
for path in '${MANIFESTS}'.split():
    into = 'corpus/training' if 'training' in path else 'corpus'  # load_corpus reads corpus/ only
    for doc_id, entry in json.load(open(path))['documents'].items():
        print(into, doc_id, entry['url'], entry['sha256'], sep='\t')
")

exit "$failed"
