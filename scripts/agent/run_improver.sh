#!/bin/zsh
# One supervised cycle of the harness-improver agent (headless Claude Code).
#   scripts/agent/run_improver.sh <panel dir, e.g. output/ab/panel1> <baseline root (a code snapshot with src/)>
# Deliverables root: $DEV_TITLES_OUTPUT (default <repo>/output).
# Works in a git worktree on branch agent/<date>; snapshots the deliverables and restores them if touched.
set -e
setopt NULL_GLOB
REPO=$(cd "$(dirname $0)/../.." && pwd); PANEL=$(cd "$1" && pwd); BASE=$(cd "$2" && pwd)
OUTPUT=${DEV_TITLES_OUTPUT:-$REPO/output}; export DEV_TITLES_OUTPUT=$OUTPUT
DATE=$(date +%Y%m%d-%H%M); WT="$REPO/../dev-titles-agent-$DATE"; OUT="$OUTPUT/agent/$DATE"
mkdir -p "$OUT"
# baseline root for spot_ab needs src/ + glossary/: use the snapshot's src with the repo's current glossary
[ -d "$BASE/glossary" ] || cp -R "$REPO/glossary" "$BASE/"
# guard: snapshot the deliverables (hashes + copies)
python3 "$REPO/scripts/agent/guard.py" snapshot "$OUT/_guard"
git -C "$REPO" worktree add -q -b "agent/$DATE" "$WT" HEAD
cd "$WT"
RUNS=$(ls -d "$PANEL"/*/ | tr '\n' ' ')
setopt NULL_GLOB  # a missing match is an empty list, not an error
P=$(basename "$PANEL"); P=${P#ab_}  # learn reports are named <panel>_<video>_<lang> (panel1_*, from ab_panel1)
LEARN=$(ls -d "$OUTPUT"/learn/${P}_* "$OUTPUT"/learn/$(basename "$PANEL")_* 2>/dev/null | tr '\n' ' ')
PROMPT="Run one cycle of the harness-improver skill (.claude/skills/harness-improver/SKILL.md). Date tag: $DATE.
Panel runs: $RUNS
Learn reports: $LEARN
Baseline root for spot_ab: $BASE
Write the report to $OUT/REPORT.md and spot-A/B output under $OUT/spot_ab."
claude -p "$PROMPT" \
  --allowedTools "Read" "Edit" "Write" "Glob" "Grep" "Bash" \
  --disallowedTools "Bash(git push:*)" "Bash(git merge:*)" "Bash(git worktree:*)" "Bash(rm -rf:*)" \
  --output-format text > "$OUT/agent.log" 2>&1 || echo "agent exited non-zero" >> "$OUT/agent.log"
# guard: any change to a deliverable is reverted and reported
python3 "$REPO/scripts/agent/guard.py" verify "$OUT/_guard" | tee -a "$OUT/REPORT.md" || true
echo "branch agent/$DATE in $WT; report $OUT/REPORT.md"
