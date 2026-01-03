#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/release.sh patch   # or minor / major
# Default: patch
BUMP_TYPE="${1:-patch}"

# Ensure clean working tree? (optional)
# If you want to allow uncommitted changes, comment this block out.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has changes (ok). Script will include them in the release commit."
fi

# 1) Pull latest main (avoid tag conflict / behind) — only if origin exists
# if git remote | grep -q '^origin$'; then
#   git fetch origin
#   git checkout master || git checkout -B master
#   git pull --rebase origin master
# else
#   echo "No remote 'origin' configured; skipping fetch/pull."
#   git checkout master || git checkout -B master
# fi

# 2) Run tests locally first (fast fail before pushing)
npm ci
npm run test --if-present
npm run lint --if-present

# 3) Bump version using npm (updates package.json + package-lock.json)
# - major/minor/patch supported
NEW_VERSION="$(node -p "require('./package.json').version")"
npm version "$BUMP_TYPE" --no-git-tag-version
NEW_VERSION="$(node -p "require('./package.json').version")"

# 4) Sync Tauri version (choose ONE method based on your config)
# ---- Method A: tauri.conf.json (Tauri v1)
if [ -f "src-tauri/tauri.conf.json" ]; then
  node -e "
    const fs=require('fs');
    const p='src-tauri/tauri.conf.json';
    const j=JSON.parse(fs.readFileSync(p,'utf8'));
    j.package = j.package || {};
    j.package.version='${NEW_VERSION}';
    fs.writeFileSync(p, JSON.stringify(j,null,2)+'\n');
  "
fi

# ---- Method B: tauri.conf.json (root)
if [ -f "tauri.conf.json" ]; then
  node -e "
    const fs=require('fs');
    const p='tauri.conf.json';
    const j=JSON.parse(fs.readFileSync(p,'utf8'));
    j.package = j.package || {};
    j.package.version='${NEW_VERSION}';
    fs.writeFileSync(p, JSON.stringify(j,null,2)+'\n');
  "
fi

# ---- Method C: Cargo.toml (Tauri Rust crate version)
# Only update if you want your src-tauri crate version to match app version.
if [ -f "src-tauri/Cargo.toml" ]; then
  node -e "
    const fs=require('fs');
    const p='src-tauri/Cargo.toml';
    let s=fs.readFileSync(p,'utf8');
    s=s.replace(/^version\\s*=\\s*\"[^\"]+\"/m, 'version = \"${NEW_VERSION}\"');
    fs.writeFileSync(p,s);
  "
fi

# 5) Commit everything (including your own changes + version bumps)
git add -A
git commit -m "release: v${NEW_VERSION}"

# 6) Create annotated tag + push main + tag
git tag -a "v${NEW_VERSION}" -m "v${NEW_VERSION}"
git push origin master
git push origin "v${NEW_VERSION}"

echo ""
echo "✅ Pushed main + tag v${NEW_VERSION}"
echo "➡️ GitHub Actions will now run tests/build and publish Release assets for v${NEW_VERSION}."
