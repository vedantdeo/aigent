#!/bin/sh
# Print the underhood commit a ref names, waiting up to an hour for underhood's own job to make it.
ref="$1"
for _ in $(seq 1 60); do
  sha=$(git ls-remote "$UNDERHOOD" "$ref" | cut -f1)
  if [ -n "$sha" ]; then
    echo "$sha"
    exit 0
  fi
  sleep 60
done
echo "::warning::underhood has no $ref after an hour" >&2
exit 1
