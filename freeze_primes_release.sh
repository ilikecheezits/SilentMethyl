#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/ocean/projects/med250012p/szhang37/SilentMethyl}"
PAPER_PATH="${2:-}"
RELEASE_DIR="releases/primes-2026-08-06"
TAG_NAME="primes-2026-final"
JOURNAL_BRANCH="journal-v2"

if [[ -z "$PAPER_PATH" ]]; then
  echo "Usage: $0 [repo-root] /path/to/SilentMethyl_PRIMES_final_2026-08-06.tex"
  exit 2
fi

if [[ ! -f "$PAPER_PATH" ]]; then
  echo "Paper not found: $PAPER_PATH"
  exit 2
fi

cd "$REPO_ROOT"
git rev-parse --is-inside-work-tree >/dev/null

echo "Repository: $REPO_ROOT"
echo "Current branch: $(git branch --show-current)"
echo "Current HEAD: $(git rev-parse HEAD)"
echo
echo "Current working-tree changes:"
git status --short
echo

read -r -p "Type FREEZE to stage all current repository changes and create the PRIMES snapshot: " answer
if [[ "$answer" != "FREEZE" ]]; then
  echo "Cancelled."
  exit 1
fi

mkdir -p "$RELEASE_DIR"
cp "$PAPER_PATH" "$RELEASE_DIR/main.tex"

if [[ -f "tcga_methylation_input_audit.json" ]]; then
  cp "tcga_methylation_input_audit.json" "$RELEASE_DIR/"
else
  echo "Warning: tcga_methylation_input_audit.json was not found in the repository root."
fi

PAPER_SHA="$(sha256sum "$RELEASE_DIR/main.tex" | awk '{print $1}')"
PRE_FREEZE_HEAD="$(git rev-parse HEAD)"

cat > "$RELEASE_DIR/RELEASE_MANIFEST.txt" <<EOF
Release: SilentMethyl PRIMES 2026
Frozen: 2026-08-06
Pre-freeze repository HEAD: $PRE_FREEZE_HEAD
Paper SHA-256: $PAPER_SHA
TCGA selected normal-sample columns: 97
TCGA total sample columns: 893
TCGA methylation matrix SHA-256: 71f7a02dd9ff849f43e05c6e54a9b8266349c9697601dc0450c9dc30f47679db
EOF

git add -A
echo
echo "Staged snapshot:"
git status --short
echo

if git diff --cached --quiet; then
  echo "No new changes to commit; tagging the current HEAD."
else
  git commit -m "Freeze PRIMES 2026 paper and analysis snapshot"
fi

FINAL_HEAD="$(git rev-parse HEAD)"

if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
  echo "Tag $TAG_NAME already exists; leaving it unchanged."
else
  git tag -a "$TAG_NAME" -m "SilentMethyl PRIMES 2026 final snapshot"
fi

if git show-ref --verify --quiet "refs/heads/$JOURNAL_BRANCH"; then
  echo "Branch $JOURNAL_BRANCH already exists."
else
  git branch "$JOURNAL_BRANCH" "$TAG_NAME"
fi

echo
echo "Snapshot complete."
echo "Frozen commit: $FINAL_HEAD"
echo "Tag: $TAG_NAME"
echo "Journal branch: $JOURNAL_BRANCH"
echo
echo "Push the saved snapshot with:"
echo "  git push origin HEAD"
echo "  git push origin $TAG_NAME"
echo "  git push origin $JOURNAL_BRANCH"
