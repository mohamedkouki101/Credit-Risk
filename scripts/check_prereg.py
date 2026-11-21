"""P-01 guard: pre-registration must come before any model code. Read-only git only.

    uv run python scripts/check_prereg.py

Fails (exit 1) if:
- a commit touches ``src/credit_risk/models/`` but the ``prereg-v1`` tag does not exist;
- the tagged commit does not contain ``PREREGISTRATION.md``;
- the first commit touching ``src/credit_risk/models/`` is not a strict descendant of the
  tagged commit (model code written before, or in, the registration commit);
- a frozen config (splits, target, features) differs from the tag and
  ``PREREGISTRATION.md`` has no ``## Amendments`` entry naming that file.

Before the tag exists and with no model code, it passes and says so. CI needs the full
history and tags (``actions/checkout`` with ``fetch-depth: 0``).
Git commands used: ``rev-parse``, ``cat-file -e``, ``log``, ``merge-base --is-ancestor``,
``diff --name-only``. None of them changes the repository.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TAG = "prereg-v1"
MODELS_DIR = "src/credit_risk/models"
PREREG_FILE = "PREREGISTRATION.md"
FROZEN = ("configs/splits.yaml", "configs/target.yaml", "configs/features.yaml")


@dataclass(frozen=True)
class GitFacts:
    tag_commit: str | None
    prereg_in_tag: bool
    first_models_commit: str | None
    tag_precedes_models: bool
    changed_frozen: list[str] = field(default_factory=list)
    prereg_text: str = ""


def evaluate(f: GitFacts) -> list[str]:
    """Return the list of violations (empty means the check passes)."""
    errors: list[str] = []
    if f.tag_commit is None:
        if f.first_models_commit is not None:
            errors.append(
                f"commit {f.first_models_commit[:7]} touches {MODELS_DIR}/ but tag {TAG} is missing"
            )
        return errors
    if not f.prereg_in_tag:
        errors.append(f"tag {TAG} does not contain {PREREG_FILE}")
    if f.first_models_commit is not None and not f.tag_precedes_models:
        errors.append(
            f"model code ({f.first_models_commit[:7]}) was committed before or with tag {TAG}"
        )
    amendments = (
        f.prereg_text.split("## Amendments", 1)[1] if "## Amendments" in f.prereg_text else ""
    )
    for path in f.changed_frozen:
        if path not in amendments:
            errors.append(f"{path} changed since {TAG} without an entry under '## Amendments'")
    return errors


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def collect_facts(root: Path) -> GitFacts:
    """Read the facts with read-only git commands."""
    tag = _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{TAG}^{{commit}}")
    tag_commit = tag.stdout.strip() or None if tag.returncode == 0 else None
    log = _git(root, "log", "--reverse", "--format=%H", "--", MODELS_DIR)
    first = log.stdout.split()[0] if log.stdout.split() else None
    prereg_path = root / PREREG_FILE
    text = prereg_path.read_text(encoding="utf-8") if prereg_path.exists() else ""
    if tag_commit is None:
        return GitFacts(None, False, first, False, [], text)
    in_tag = _git(root, "cat-file", "-e", f"{tag_commit}:{PREREG_FILE}").returncode == 0
    precedes = (
        first is not None
        and first != tag_commit
        and _git(root, "merge-base", "--is-ancestor", tag_commit, first).returncode == 0
    )
    diff = _git(root, "diff", "--name-only", tag_commit, "--", *FROZEN)
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    return GitFacts(tag_commit, in_tag, first, precedes, changed, text)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    facts = collect_facts(root)
    errors = evaluate(facts)
    for e in errors:
        print(f"check_prereg: FAIL: {e}", file=sys.stderr)
    if errors:
        return 1
    if facts.tag_commit is None:
        print(f"check_prereg: OK (tag {TAG} not created yet; no model code committed)")
    else:
        print(f"check_prereg: OK (tag {TAG} at {facts.tag_commit[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
