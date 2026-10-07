#!/bin/sh
# Print the underhood commit a tag names, waiting up to an hour for underhood's own job to make it.
tag="$1"
for _ in $(seq 1 60); do
  sha=$(git ls-remote "$UNDERHOOD" "refs/tags/$tag" | cut -f1)
  if [ -n "$sha" ]; then
    echo "$sha"
    exit 0
  fi
  sleep 60
done
echo "::error::underhood has no $tag after an hour" >&2
exit 1
