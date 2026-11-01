# About this document

This is the **master record** of the project "The Babel Effect": one long file that holds everything we did, formal and informal, so that nothing lives only in a chat or in someone's head. It is the go-to document. The short paper is one chapter of it (Part 5). It is written to be revised: change a source file, rerun one command, get a new PDF.

**Written:** 2026-11-01 (your clock, IST). **State of the work at writing time:** Llama-3.2-3B-Instruct finished on Kaggle and in the paper; Gemma-2-2B running on Kaggle (fourth attempt), no Gemma result read yet; Llama-3.2-3B base and four more models not started.

**Time zones.** Kaggle console times are UTC. Your clock is IST, which is UTC plus 5 hours 30 minutes. Every time in this record says which one it is, or is a date only.

## How this record is built (so you can revise it)

| What | Where | How to change it |
|---|---|---|
| Chapters written by hand (Parts 1 to 4, 6, 7, 9) | `paper/master/*.md` | Edit the file, rebuild. |
| The paper (Part 5) | `paper/paper_v2_draft.md` | Edit the paper; everything here follows. |
| Master goal and plan file (copied into Part 10) | `GOAL.md` | Edit `GOAL.md`. |
| Pre-registrations, decision log, experiment log, plans, external review (Parts 8 to 10) | `docs/*.md` | Edit those files; they are copied in word for word. |
| Lists of paths, run folders and git history (Part 9) | generated | Rebuilt from the repository each time. |

**Rebuild:** `.venv/Scripts/python.exe paper/build_master.py` from the repo root. Needs `pandoc` and Microsoft Edge. Output: `paper/The_Babel_Effect_MASTER_RECORD.pdf` (and the `.md` it was built from).

## What is checked and what is not

- The numbers in the paper chapter (Part 5) are checked against files by `results/manifest.json` (2126 claims verified on 2026-11-01 before the first commit of this record).
- Numbers in the hand-written chapters are copied from the files named next to them (`docs/HYPOTHESIS.md`, `GOAL.md`, `docs/LOG.md`). They are not machine-checked; if one disagrees with the paper chapter, the paper chapter and the files win.
- Literature statements come from paper abstracts and retrieved passages, not full reads, unless the text says otherwise. Venues were not checked. Papers are cited by arXiv identifier.
- Copied documents (Parts 8 to 10) are unedited. They contain run codes (A1 to A30) and many dates; the chapters before them give the plain-words version.

## Contents

1. **Part 1. Goal, rules and what we are trying to prove.** The founding brief, how the goal changed, the standing rules.
2. **Part 2. The story in order.** What we did from 2026-10-12 to 2026-11-01, with the key numbers.
3. **Part 3. Problems faced.** Every setback we hit, its cause, the fix, and whether it is closed.
4. **Part 4. In your words, and my mistakes.** The instructions you gave in your own words, and the errors I made, stated plainly.
5. **Part 5. The paper.** Current draft, word for word.
6. **Part 6. Results at a glance, and what decides the paper.** One page of numbers.
7. **Part 7. Open items and next steps.** Including what to do when the Gemma Output arrives.
8. **Part 8. Pre-registrations.** `docs/HYPOTHESIS.md`, word for word.
9. **Part 9. Where everything is.** Paths, commands, notebooks, run folders, git history.
10. **Part 10. Logs and plans.** `docs/LOG.md`, `docs/EXPERIMENT_LOG.md`, `docs/NEXT.md`, `docs/ROADMAP_Q1.md`, the master goal file, the external review of 2026-10-23, and the draft of the control analysis (A12a).
