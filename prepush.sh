#!/usr/bin/env bash
# Pre-push checks. Run from the repo root:  ./prepush.sh
#
# Hard checks exit non-zero. Advisory checks print and keep going, because
# they are judgement calls rather than errors.

set -uo pipefail
cd "$(dirname "$0")"
fail=0

section() { printf '\n=== %s ===\n' "$1"; }
ok()      { printf '  ok    %s\n' "$1"; }
bad()     { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }
note()    { printf '  note  %s\n' "$1"; }

section "tests"
if python3 -m pytest -q >/tmp/pytest.log 2>&1; then
  ok "$(tail -1 /tmp/pytest.log)"
else
  bad "pytest failed; see /tmp/pytest.log"
  tail -20 /tmp/pytest.log
fi

section "lint"
if ruff check . >/tmp/ruff.log 2>&1; then ok "ruff clean"; else bad "ruff errors"; cat /tmp/ruff.log; fi

section "README numbers match committed results"
if python3 scripts/check_readme_claims.py; then :; else bad "README/results mismatch"; fi

section "package imports from a clean interpreter"
if python3 -c "import uplift; assert uplift.__version__" 2>/dev/null; then
  ok "import uplift"
else
  bad "package does not import"
fi

section "placeholders left in prose"
hits=$(grep -rn "TODO\|FIXME\|XXX\|___\|<fill\|in progress\|lorem" README.md docs/ 2>/dev/null | grep -v "^docs/DECISIONS.md.*Open" || true)
if [ -z "$hits" ]; then ok "none"; else bad "placeholders found"; echo "$hits"; fi

section "data and artefacts not committed"
tracked_data=$(git ls-files 'data/raw/*' | grep -v '.gitkeep' || true)
if [ -z "$tracked_data" ]; then ok "data/raw is empty but for .gitkeep"; else bad "dataset tracked"; echo "$tracked_data"; fi
big=$(git ls-files -z | xargs -0 -I{} sh -c 'test -f "{}" && [ $(wc -c <"{}") -gt 1048576 ] && echo "{}"' 2>/dev/null || true)
if [ -z "$big" ]; then ok "no tracked file over 1MB"; else bad "large files tracked"; echo "$big"; fi

section "stray files"
stray=$(find . -path ./.git -prune -o -name '.DS_Store' -print 2>/dev/null | head -10 || true)
if [ -z "$stray" ]; then
  ok "no .DS_Store"
else
  note "gitignored, but tidy up: find . -name '.DS_Store' -delete"
  echo "$stray"
fi

section "secrets"
secrets=$(git grep -nIE '(api[_-]?key|secret|password|token)[[:space:]]*=[[:space:]]*["'"'"'][^"'"'"']{8,}' -- . 2>/dev/null || true)
if [ -z "$secrets" ]; then ok "no literal credentials"; else bad "possible credential"; echo "$secrets"; fi

section "commit attribution"
authors=$(git log --format='%an <%ae>' | sort -u)
printf '  authors: %s\n' "$(echo "$authors" | tr '\n' ' ')"
[ "$(echo "$authors" | wc -l)" -eq 1 ] && ok "single author" || note "more than one author"
ai=$(git rev-list HEAD | while read -r h; do
  git log -1 --format='%B' "$h" | grep -qi "claude\|copilot\|chatgpt\|generated with" && echo x
done | wc -l | tr -d ' ')
total=$(git rev-list --count HEAD)
if [ "$ai" -eq 0 ]; then
  ok "no AI references in commit messages ($total commits)"
else
  note "$ai of $total commit(s) reference an AI tool — your call whether to keep"
fi

section "branch matches CI"
ci_branch=$(grep -A2 '^on:' .github/workflows/ci.yml | grep -o 'main\|master' | head -1)
if [ "$(git rev-parse --abbrev-ref HEAD)" = "$ci_branch" ]; then
  ok "branch and CI trigger agree ($ci_branch)"
else
  bad "on branch $(git rev-parse --abbrev-ref HEAD) but CI triggers on $ci_branch"
fi

section "git state"
git status --short
branch=$(git rev-parse --abbrev-ref HEAD)
printf '  branch: %s\n' "$branch"
git remote -v | head -2

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'All hard checks passed. Clear to push.\n'
else
  printf '%d hard check(s) failed. Fix before pushing.\n' "$fail"
fi
exit "$fail"
