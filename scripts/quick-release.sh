#!/usr/bin/env bash
set -euo pipefail

# Release nhẹ: commit message tùy ý, không bump version, push + tag (tùy chọn)
# Usage:
#   bash scripts/quick-release.sh "chore: misc"            # chỉ commit + push
#   TAG=v0.1.2 bash scripts/quick-release.sh "release msg" # commit + push + tag + push tag

MSG="${1:-chore: misc updates}"
TAG="${TAG:-}"

git add -A
git commit -m "$MSG" || echo "Nothing to commit"
git push origin HEAD || true

if [ -n "$TAG" ]; then
  git tag "$TAG" || echo "Tag exists?"
  git push origin "$TAG" || true
fi

echo "Done. Commit: $MSG; Tag: ${TAG:-none}"
