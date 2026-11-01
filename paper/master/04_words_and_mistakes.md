# Part 4. In your words, and my mistakes

This part keeps the informal side of the project: what you asked for in your own words (typed as you typed them, spelling kept), how the requests came, and the errors I made, stated plainly. Nothing here is edited to sound better.

## 4.1 Your instructions, in your words

| Date | You wrote | What it changed |
|---|---|---|
| 2026-10-25 | "we will do all the work, the conference we will decide later to which one we should submit... q1 or q3 or q2 journal that's the goal" | Journal target; no venue talk. |
| 2026-10-26 | "target q1 plan accordingly" and "it is about the work, not the paper; if the work is good we write the paper as good as we can" | Q1 roadmap; science first. |
| 2026-10-29 | "dont use A12 A33 i dont like it make it easier so i will understand" | No run codes in chat. |
| 2026-10-29 | "SHOULDNT HAVE ANY ERRORS ... CONSUME ALL OF THE KAGGLE GPU" | Zero errors; both GPUs busy. |
| 2026-10-29 | One unified notebook per model that runs the checks first, then everything; you run it, download the Output, send it back. No GitHub inside the notebook; all code packed inside. | `kaggle/make_notebook.py`. |
| 2026-10-30 | "forget about our laptop run, dont bring it into the picture, focus on the Kaggle run, make it the best" and "if we get extra results make them best" | Kaggle-only evidence; strengthen a new result before claiming it. |
| 2026-10-31 | "whatever u give me I will run in the Kaggle, we will only get accepted if we have novelty, that's it." | Every notebook judged by what is new. |
| 2026-10-31 | "so we are the first doing work of this kind?" | My answer: no for the field; "not found in prior work" only for the per-fact key test. |
| 2026-10-31 | Use the models the published papers used; add baselines and a benchmark slice. | Model order and the plan in `GOAL.md`. |
| 2026-10-31 | "ok...SEE BY THIS TIME U SHOULD HAVE UNDEESTAND HOW CODE SHOULDNT HAVE ERRORS OMGG" | Dry-run rule (Part 1, rule 5). I said "go" was needed for GPU use; you gave it ("go"). |
| 2026-10-31 | "can i close munderdufflin i mean u" | My answer: no, the background run is a child of this session. |
| 2026-10-31 | "give in one line", "DATASET LINK", "CAN I RUN IN ANOTHER LAPOTP BUT SAME ACCOUNT?", "BRO SORRY TELL ME WHAT TO DO", "OR GIVE ME DATAET LINK", "NOTEBOOK PATHH" | Short answers; the notebook path every time it changes; the Kaggle Model route instead of a dataset. |
| 2026-11-01 | "NOW ITS 2:50 WHEN WILL THIS BE DONE?" | I had given UTC times without saying so. Now always UTC or IST. |
| 2026-11-01 | "I HAVE SENT U THE PATHS OF THE NOTEBOOK AND REUSLTS WRITE KEEP THEM IN ONE SINGLE FOLDER ORGANZE WHATEEVER WE DID EVERYTHING UNITL NOW" | `BABEL_EFFECT_ORGANIZED\` (a copy, 69 MB). |
| 2026-11-01 | "push and commit to the github" | Commits and push (the standing rule was: only when you say so). |
| 2026-11-01 | "now write eeveeyrhitng into the paper ...so that we are documenting everythign its okay document each and eerythign formal informal into the paper we can revise it make it bettwe lter in the same paper ui can also put paths we are assuming it as our go to huge pdf file which says eeeverytihg about the paper include goal and probelms faced everything too" | This master record. |
| 2026-11-01 | "COMMIT AND PUSH TO GITHUB DONT BE LATE..." | Pushed the earlier commits at once; this record is a later commit. |

## 4.2 How we worked

- You run the notebook on Kaggle and send the result back: a pasted console, a saved console notebook from `Downloads`, or the Output zip. I read the files, I do not see Kaggle live.
- Questions came as short messages, often in capitals when something was wrong or urgent. The pattern: a status question ("how much time left?"), a path request ("NOTEBOOK PATHH"), a worry ("does this notebook have any erorrs like the previous one?"). I answer with numbers first and one decision.
- You prefer fewer round trips: one message per task, and the notebook doing its own checks.
- You said "go" before each laptop GPU run (rule: I tell you first and wait).
- You are the first author. The acceptance goal drives every choice: novelty first, then rigour.

## 4.3 My mistakes, plainly

1. **Wrong smoke-test prompt for the Gemma base model.** I had not checked it; the first Gemma attempt died at the smoke job after 7.6 minutes. Fixed with a question-style prompt.
2. **Handed over a notebook that failed on the real model, twice.** My checks covered scripts, flags and unit tests, not a real run of each job type. After your anger I adopted the dry-run rule and ran every Gemma job type on the laptop GPU.
3. **A wrong theory about the cause.** I wrote that identical subject names were ruled out because Llama had 24 such facts. The property that matters is an identical key, not an identical name. Corrected in the amendment text before the final notebook.
4. **Wrong time guess.** I guessed tracing at 30 minutes; it took 108.
5. **Wrong prediction in the science.** I predicted that shared noise inflated the EN-ES alignment; it did not (except for Gemma equal-span facts).
6. **UTC without saying so.** My "expected end" time was in UTC while you read it as your own clock. Now always labelled.
7. **A fragile self-test.** A timing test I wrote failed twice inside the Llama notebook folder; since a failing self-test stops the whole Kaggle run, that would have cost a run. Widened.
8. **Harness mistakes during the dry runs.** Two bad fact files (no answer pool; too few valid facts for the controls) cost time before I used the real fact list.
9. **An early pre-run check missed data files.** The first A28 Llama run lost two Spanish analysis jobs because a run folder was missing from the layout list. My check had covered scripts and flags only.
10. **Reading a shortcut too quickly.** The Kaggle dataset you suggested for Gemma was Gemma 1; I checked file sizes against the maker's repo and refused it.

## 4.4 Things that are true but easy to forget

- Nothing in the A28 result is a claim about a better editor: the shared-key fix was null.
- The Hindi layer-controlled result passes by a hair (lower end 0.011). It is a narrow pass, not a clean one.
- All strong numbers are from one model so far.
- The laptop experiments and the Kaggle experiments are different evidence; the paper keeps them apart (Section 4.8 is Kaggle only).
- The Gemma run that is running now was started from a notebook whose key-similarity, key-mode and graph jobs have not yet run on Gemma data anywhere.
