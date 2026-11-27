# Provenance

This repository is a public copy of selected files from two private repositories. It has fresh history because the private histories contain account details and trade results.

| Part | Source revision | Private history at that revision |
|---|---|---|
| `pine`, `tools`, `backtest` | `6bb32d9`, committed 2026-10-01 | 89 commits from 2026-08-20 |
| `scalping` | `c525f9d`, committed 2026-08-17 | 151 commits from 2026-06-17, covering other work as well |

Files were read from the committed revisions with `git archive`, then passed through a scripted allowlist, redaction and deny-list check.

## Files written for this copy

- `README.md`, `PROVENANCE.md`, `.gitignore`, `.github/workflows/ci.yml`, `scalping/pyproject.toml`
- `docs/images/trade-anatomy.svg`, a schematic with invented bars
- `docs/images/backtest-august.webp` and `docs/images/ccl-replay.webp`, screenshots supplied by the author

Everything else is project code, data or project documentation.

## Left out

- Trade lists, result tables, rosters that carry per-name results, and the scripts that only render them
- Press releases (public, and re-fetchable with `tools/earnings-now.py`)
- Intermediate Pine versions between the eight milestones
- Vendor calendar data, local databases, agent instruction files and internal notes
- Execution, broker and account code (the SQLite store keeps its order and trade tables)

## Changed

- In code comments and tooltips, performance figures are replaced with `[n]`: win rates, profit factors, expectancies, returns, sample sizes and R results. Counts of test events, signals and symbols in comments are treated the same way. Mechanism parameters such as thresholds and rung levels are unchanged in code. Where a comment or tooltip states an R level in the same passage as results, that R level is masked as well.
- The README's Backtest results section reports two measured runs that the author chose to publish, with links to the full reports. `docs/images` holds a screenshot of the August report and a bar-replay chart.
- Home-directory paths, email addresses and the SEC contact string are replaced with environment variables or placeholders.
- Files moved: `docs/backtest` to `backtest`, milestone files to `pine/history`, and the scalping sources under `scalping`. Docstrings follow the new paths.
- Quoted remarks and the author's first name are removed from comments, docstrings and tooltips, which shortens the Pine files. Executable lines are unchanged.
- Some comments cite internal documents and scripts that are not published.
- The scalping indicator keeps its internal title, "Lane A Cockpit".
- Em dashes in code comments are normalised to hyphens. Pine tooltips keep their punctuation because they are part of the indicator's interface.

Every Python file was compared with its original syntax tree. All but five match apart from literals. In `tools/earnings-now.py` and `tools/score_extraction.py` a path and a contact string became environment variables, and in `tools/score_extraction.py`, `tools/build_corpus.py`, `tools/atr_at_entry.py` and `backtest/recover_missing.py` a path changed because the files moved.

## Verification

`python -m pytest tests` in `scalping`, byte-compilation of `tools` and `backtest`, and `tools/check.py` on the indicator were run on this copy.
