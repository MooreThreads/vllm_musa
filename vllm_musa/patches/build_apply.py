# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""build-time patch-series applier.

Applies the vLLM-MUSA unified-diff series to the cloned ``third_party/vllm@<tag>``
at **build time** via ``git apply``, idempotently — so the installed vLLM is
*pre-patched*. This is **the** vLLM source-patch mechanism; there is no runtime
source-patching and no fallback. (Live-object monkey-patches that have no
source-diff form — the spec-decode kernel prime and the draft-TP=1 wiring — run
at import time via ``apply_object_patches``; they are not source patches.)

Idempotency: a patch that already
``git apply --reverse --check``-es is skipped; otherwise it is applied forward;
a forward failure is a loud conflict (the pinned vLLM moved — regenerate the
series). Standard ``git apply`` (no fuzz); for true 3-way merge across a version
bump use the ``git format-patch``/``git am -3`` maintenance flow (a real git
checkout), not this build applier.

**Stdlib only, no ``vllm_musa`` package import** — ``setup.py`` loads this by
file path (``importlib``) before ``vllm_musa`` is installed, so importing it must
not trigger ``vllm_musa/__init__.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def apply_patch(
    repo: Path, patch: Path, strip: int = 1, check_only: bool = False
) -> str:
    """Idempotently apply one unified-diff ``patch`` to ``repo``.

    Returns ``"already-applied"`` (reverse-applies cleanly → skip),
    ``"applied"`` (forward-applied now), or ``"conflict"`` (does not apply →
    the caller should fail loud: the pinned vLLM source drifted).

    With ``check_only`` the patch is NOT applied: returns ``"already-applied"``,
    ``"would-apply"`` (forward-checks clean), or ``"conflict"``. This is the
    non-mutating dry run used by ``musa_sync verify``.
    """
    # git -C changes repository context but not the process cwd used to
    # resolve patch filenames. Build invokes this helper with paths relative
    # to the vLLM-MUSA root, so pass an absolute patch path to git.
    patch = patch.resolve()
    p = ["--recount", "-p", str(strip)]
    # Already applied? (the reverse patch applies cleanly to the current tree.)
    if _git(repo, ["apply", "--reverse", "--check", *p, str(patch)]).returncode == 0:
        return "already-applied"
    # Will it apply forward cleanly? (dry run — loud, located on failure.)
    if _git(repo, ["apply", "--check", *p, str(patch)]).returncode != 0:
        return "conflict"
    if check_only:
        return "would-apply"
    r = _git(repo, ["apply", "--whitespace=nowarn", *p, str(patch)])
    return "applied" if r.returncode == 0 else "conflict"


def series_files(series_dir: Path) -> list[Path]:
    """Ordered series. A quilt-style ``series`` manifest wins (one patch name per
    line, ``#`` comments ignored); otherwise sorted ``*.patch`` glob."""
    manifest = series_dir / "series"
    if manifest.is_file():
        out = []
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(series_dir / line)
        return out
    return sorted(series_dir.glob("*.patch"))


def apply_patch_series(
    repo: Path,
    series_dir: Path | None = None,
    strict: bool = True,
    check_only: bool = False,
    order: list[Path] | None = None,
) -> list[tuple[str, str]]:
    """Apply (or, with ``check_only``, dry-run) the series to ``repo``.

    Returns ``[(patch_name, status)]``. Raises ``RuntimeError`` on the first
    ``conflict`` when ``strict`` (the default — a drifted required patch must
    fail the build, not silently skip).

    Patch order: if ``order`` (an explicit list of ``.patch`` paths — e.g. the
    MDM manifest filtered to one ``apply_phase``, in apply order) is given it is
    used verbatim; this is how ``apply_phase`` ordering is honored.
    Otherwise the series is discovered from ``series_dir`` (quilt ``series``
    manifest or sorted glob). A missing ``series_dir`` with no ``order`` is a
    no-op (returns ``[]``), so the build is safe before any patch is migrated.
    """
    results: list[tuple[str, str]] = []
    if order is not None:
        patches = [Path(p) for p in order]
    elif series_dir is not None and Path(series_dir).is_dir():
        patches = series_files(Path(series_dir))
    else:
        return results
    for patch in patches:
        status = apply_patch(repo, patch, check_only=check_only)
        results.append((patch.name, status))
        if status == "conflict" and strict:
            raise RuntimeError(
                f"MUSA build patch {patch.name!r} does not apply to {repo} — the "
                f"pinned vLLM version likely moved. Regenerate the series against "
                f"the new tag."
            )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply the vLLM-MUSA build-time patch series."
    )
    ap.add_argument("repo", help="path to the cloned vLLM checkout (a git repo)")
    ap.add_argument(
        "series_dir", help="dir of .patch files (+ optional `series` manifest)"
    )
    ap.add_argument(
        "--no-strict", action="store_true", help="warn instead of failing on conflict"
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="dry-run: report status without applying",
    )
    a = ap.parse_args(argv)
    results = apply_patch_series(
        Path(a.repo),
        Path(a.series_dir),
        strict=not a.no_strict,
        check_only=a.check_only,
    )
    for name, status in results:
        print(f"{status:16} {name}")
    n_conflict = sum(1 for _, s in results if s == "conflict")
    n_applied = sum(1 for _, s in results if s in ("applied", "would-apply"))
    n_skip = sum(1 for _, s in results if s == "already-applied")
    verb = "would-apply" if a.check_only else "applied"
    print(
        f"--- {n_applied} {verb}, {n_skip} already-applied, {n_conflict} conflict ---"
    )
    return 1 if n_conflict else 0


if __name__ == "__main__":
    sys.exit(main())
