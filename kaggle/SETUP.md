# Running the Babel Effect runs on Kaggle (free GPUs)

What you do: steps 1-5 (about 15 minutes, once). What the notebook does: everything else.

1. **Account.** Sign in at kaggle.com. Settings > Phone verification > Verify now. Without this you get no GPU.
2. **Hugging Face token.** On huggingface.co, accept the Llama-3.2-3B-Instruct licence, then make a read token (Settings > Access Tokens).
   On Kaggle: Add-ons > Secrets > add a secret named `HF_TOKEN` with that token. Never paste the token into a cell.
3. **Upload the bundle.** Kaggle > Create > New Dataset > upload `kaggle/babel_bundle.zip` (25 MB). Name it `babel-bundle`. Keep it **Private**.
4. **Import the notebook.** Create > New Notebook > File > Import Notebook > pick `kaggle/babel_kaggle.ipynb`.
5. **Settings (right side).** Accelerator: **GPU T4 x2**. Internet: **On**. Add Data: your `babel-bundle` dataset. Notebook sharing: Private.

## First session (paraphrase check + new-facts run, both at once)
- Leave `STAGE = "first"`. First run the cells one by one up to "Tests and a quick model check" (about 10 minutes). If a cell fails, stop and send me the error text.
- If all is fine: **Save Version > Save & Run All**. It then runs by itself (your laptop can be off). Expect 3-6 hours, depending on how fast the T4 is.
- When it ends, open the notebook's Output tab and download `babel_new_results.zip`. Unzip it into `babel-effect/analysis/` and tell me.

## Second session (shared-edit test)
- Make a new version of the notebook with `STAGE = "second"`. In Add Data, add the **output of the first session** as an input. Save & Run All. Expect 6-10 hours.
- Download `babel_new_results.zip` the same way.

## If a session is cut off
Just run the notebook again with the same stage. The scripts skip finished steps and resume the edit run where it stopped.

## Known unknowns (not yet tested on Kaggle)
- The T4 has no native bf16 (the 16-bit number format our code uses). It should still run, but maybe slower. The quick check in step 4 of the notebook prints the speed.
- Exact package versions may clash with Kaggle's image. The notebook falls back to the key packages by itself.
- The weekly GPU limit (reported as about 30 hours) is from forum posts, not Kaggle's docs. Check Settings > Quotas on your account.
