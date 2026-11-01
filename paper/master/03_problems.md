# Part 3. Problems faced

Every setback we hit, what caused it, what we did, and whether it is closed. "Mine" marks an error I made. Details and exact numbers are in the decision log (Part 10) and the pre-registrations (Part 8).

## 3.1 Set-up and hardware

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 1 | Llama and Gemma could not be downloaded | Gated on Hugging Face; the stock downloader stalled at 0 bytes on this network | Access requested; chunked resumable downloader `code/dl_hf.py` (about 0.5 MB/s); GPT-2 used only as a pipeline smoke test | Closed. Later runs use Kaggle Models, no token |
| 2 | PyTorch had no GPU support | The RTX 5060 (Blackwell) needs the CUDA 12.8 wheels | Installed from the cu128 index | Closed |
| 3 | TransformerLens would not import `HookedTransformer` | Version 4.x removed it | Pinned `transformer-lens==3.9.0` | Closed |
| 4 | Memory blow-up on the 8 GB GPU | One forward pass peaked at 7.49 GiB; TransformerLens copies weight matrices on every pass and autograd kept about 1.2 GiB alive per pass, spilling a debug run to 14.5 GiB in shared memory | Weights laid out so the copies are not kept; caches freed per fact; chunked sweeps (details in the log, 2026-10-12 to 14) | Closed |
| 5 | The layer sweep died twice | A reboot (10-15) and Windows Modern Standby (10-18) | Resumable sweeps; `keep_awake.ps1`, watchdog and queue scripts | Closed |
| 6 | Weights provenance unclear | The laptop copy came from the Unsloth mirror | SHA-256 matched Meta's LFS objects; config and tokenizer files differ from Meta's, so the small files from the repo are used in the notebooks | Closed |

## 3.2 Science and statistics

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 7 | Token counts differ by language | Median subject length 2 / 3 / 5 tokens in EN / ES / HI | Span check (`code/check_spans.py`): a contiguous subject span for every kept fact and language, 0 errors | Closed |
| 8 | Outside review: "reject" for the main venues | Missing same-language, centred and lexical-overlap controls; shared noise seed across languages; unnormalised profiles; a Spearman tie bug | Controls registered before computing (A12, A12a); the review's one false premise (46 identical-prefix facts) checked and rejected (real number: 1) | Closed |
| 9 | My prediction was wrong (mine) | I predicted shared noise inflated the EN-ES alignment | It matters only for Gemma equal-span facts; stated in the log | Closed |
| 10 | In-context baseline anomaly | Llama English statement fails in English (0/10) but works in another form | Diagnosed and written up (EXPERIMENT_LOG, 2026-10-25) | Closed |
| 11 | Hindi underpowered | 23 and 12 source entities, below the floor of 25 | Larger fact set (A16) | Closed |
| 12 | Last-token Hindi result looked low (0.74 to 0.76) | Hindi subjects span several tokens | Whole-subject (span-wise) version (A18): 0.86 / 0.87 Llama, 0.93 / 0.81 Gemma | Closed |
| 13 | The "name overlap explains it" reading | Not true for Llama | Revised after A22; paper Section 4.6 rewritten | Closed |
| 14 | Tracing does not predict which individual facts are shared | rho = -0.04 | Reported as a result, not hidden | Closed (a finding) |
| 15 | Laptop and Kaggle single edits differ | Median difference 0.10, no median shift (fp16 on T4 vs laptop) | Rule: compare Kaggle with Kaggle only; the notebook traces the facts itself | Closed |
| 16 | The shared-key edit gave no gain | It does not fix transfer (best gain 0.044) | Reported as a null; the diagnostic paper (Plan B) stands | Closed (a finding) |
| 17 | Non-geography facts untestable | Only two relations survive (official_language, book_author); 53 EN-ES and 13 EN-HI facts | Needs more relations, registered first | **Open** |
| 18 | The Hindi result is narrow | Lower end of the layer-controlled interval 0.011; layer-map pooled value 0.17 misses the bar | Needs a second model | **Open** (Gemma running) |
| 19 | Novelty is thin | Cross-lingual edit failure, shared representations and orthogonal edit vectors are already published | One candidate novelty: per-fact key similarity vs transfer; worded as "not found in prior work" | **Open** (five closest papers still to be read in full) |
| 20 | Hindi and Spanish facts not checked by a native speaker | No checker | `data/audit/native_speaker_audit.csv` (362 facts) is ready | **Open** (needs a Hindi and a Spanish checker) |

## 3.3 The Kaggle notebooks

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 21 | First notebook crashed | Numpy upgraded under a live kernel | `kaggle/driver.py` runs every job in its own process (stdlib until pip, 3 attempts, resume) | Closed |
| 22 | Config and tokenizer files differ from Meta's | Laptop copy from a mirror | The notebook carries the laptop's small files | Closed |
| 23 | Two Spanish non-geography jobs failed 3 of 3 (Llama run, 10-30) | A run folder was missing from `kaggle/layout.py` | Fixed; the notebook was then checked unpacked in a clean folder; those jobs passed in the next run | Closed. My pre-run check covered scripts and flags, not the data files each job reads (mine) |
| 24 | Tracing took 108 min (guess 30); GPU 1 idle for 73 min | My estimate was off; tracing runs on one GPU | Re-estimated: whole run 348 min | **Open**: split tracing across both GPUs for later models |
| 25 | Progress bars count to 200 although 87 facts qualify | Bar total is the cap, not the number that qualifies | Noted | Cosmetic |
| 26 | Unit tests passed inside the repo but failed inside the notebook | Tests read repo-only files | Those tests skip inside a notebook; the notebook's own tests are now run unpacked after every rebuild | Closed |
| 27 | A timing self-test failed twice inside the Llama notebook folder (mine) | Margins too tight (1.17 vs 1.4) | Widened; a failing self-test stops the whole Kaggle run, so timing tests need wide margins | Closed |

## 3.4 Gemma-2-2B

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 28 | Smoke job failed 3 times | A base model completes "The capital of France is" with " a" (mine: I had not checked the base model with this prompt) | Question-style prompt; smoke prints the top five tokens on a wrong answer; fp16 tested under tracing noise | Closed |
| 29 | Shared-key edit failed 3 of 3 | Identical EN/ES keys (cosine 1.0) make the joint-edit matrix singular | Identical keys kept once; solve in float64; later key modes may fail without stopping the job; failure text visible in the console | Closed in code; not yet seen on the GPU in Kaggle |
| 30 | My first theory was wrong (mine) | I wrote that identical subject names were ruled out because Llama had 24 such facts with key cosines below 0.995. What matters is an identical key (same subject text and same left context), not the same name | The amendment text was corrected before the final notebook | Closed |
| 31 | The failure had no traceback in the console | The driver hid the job log | Driver prints the last 14 log lines of a failed job | Closed |
| 32 | I handed over a notebook that failed twice on the real model (mine) | Unit tests and flag checks do not run the real model | Rule: dry-run every new model's job types on the laptop GPU first (done for Gemma, 4 facts) | Closed (rule in force) |
| 33 | My dry-run harness failed (mine) | A one-fact file left no answer pool; a three-fact file left too few valid facts for controls | Harness `run_one_fact.py` patches the fact selection and uses the full trace file | Closed |
| 34 | Base-edit dry run killed | The system ended it for low memory after 1 of 8 edits | Not restarted (standing rule) | **Open** (optional: rerun, about 25 min, other programs closed) |
| 35 | Third attempt stopped after 1 minute | The Gemma Kaggle Model was not attached | Notebook says so plainly | Closed (setup) |
| 36 | A suggested dataset was the wrong model | `leimeng46/gemma-2-2b` is Gemma 1 (18 layers) | Notebook refuses it; Google's own Kaggle Model is used | Closed |
| 37 | Key-similarity, key-mode and graph jobs never run on Gemma data | Attempt 2 stopped before them; attempt 4 not finished | Run on the real Output when it arrives; CPU-only, can be rerun locally | **Open** |

## 3.5 Communication and process

| # | Problem | Fix |
|---|---|---|
| 38 | You were angry about errors ("how code shouldnt have errors"). | Zero-error rule; dry runs before hand-over (Part 1, rule 5). |
| 39 | Run codes in my replies confused you. | Plain names in chat; codes stay in files. |
| 40 | I stated Kaggle times in UTC without saying so, so "NOW ITS 2:50" did not match. (mine) | Always say UTC or IST. |
| 41 | A detached background subshell made a task notification arrive early. (mine) | Run the process directly in the background. |
| 42 | Closing the assistant app would stop the background run, because it is a child of the session. | Told you not to; the Kaggle run itself is unaffected by closing the laptop app, but my laptop jobs are. |
| 43 | The Kaggle weekly GPU quota is not confirmed (about 30 hours is a guess). | **Open**: check on Kaggle before the next models. |
| 44 | A Hugging Face token was used earlier. | **Open**: revoke it. |
