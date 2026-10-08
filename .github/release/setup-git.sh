#!/bin/sh
# Let git clone underhood with UNDERHOOD_TOKEN, commit as the bot, and push here as $AIGENT.
set -eu
git config --global url."https://x-access-token:${UNDERHOOD_TOKEN}@github.com/vedantdeo/underhood".insteadOf "$UNDERHOOD"
git config --global user.name "github-actions[bot]"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
echo "AIGENT=https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}" >> "$GITHUB_ENV"
