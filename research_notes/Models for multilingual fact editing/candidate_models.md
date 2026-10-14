# Candidate small open-weight LLMs to ADD to a cross-lingual (EN/ES/HI) factual-recall and knowledge-editing study (8 GB VRAM)

Date of research: 2026-10-14. Conventions: "config" = fetched from the model's config.json on Hugging Face (HF). "Computed" = my arithmetic from config values (bf16 = 2 bytes/param, GiB = 2^30); the computed totals matched the card values wherever a card gave one (Qwen2.5-3B 3.09B, Qwen3-4B 4.0B, Phi-4-mini 3.8B, Granite-3.3-2B 2.5B). "Own measurement" = tokenizer fertility I ran locally (method below). UNVERIFIED = not confirmed from a primary source.

Baseline for budget: the user reports Llama-3.2-3B bf16 weights are about 5.98 GiB (computed) and peak at about 7.5 GiB in their setup, so roughly 1.5 GiB of headroom over weights is the working rule. Anything with bf16 weights above about 6.2 GiB is unlikely to fit in bf16 (inference of that size may work, autograd/editing will not).

---

## 1. Architecture, size, bf16 footprint, languages, license, gating

### Takeaway
Only models with bf16 weights under about 6 GiB fit with Llama-3.2-3B-like headroom: Qwen3.5-2B, Qwen3-1.7B, Gemma-2-2B, Gemma-3-1B, SmolLM3-3B, Sarvam-1, Granite-3.3-2B, Llama-3.2-1B and Qwen2.5-1.5B/3B. Qwen3-4B, Qwen3.5-4B, Phi-4-mini, Gemma-3-4B (even text-only) and Gemma-4 E2B/E4B do not fit in bf16 and would need 4-bit. Of the models officially listing both Hindi and Spanish, only Llama-3.2 (the baseline), Tiny Aya, Qwen3/Qwen3.5 and Gemma-3/4 (via "140+ languages") qualify; Qwen2.5, Phi-4-mini, SmolLM3 and Granite-3.3 list Spanish but not Hindi, and Sarvam-1 lists Hindi but not Spanish.

### Cited Findings

Table (config values from each model's config.json; "tied" = tied input/output embeddings; weights GiB are computed for bf16 and include the embedding matrix once if tied):

| Model | Params (card) | Layers | d_model | Vocab | bf16 weights (computed) | Fits 8 GB bf16 w/ autograd headroom? | Hindi listed? | Spanish listed? | Base / Instruct | License / gating |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct (baseline) | 3.2B | 28 | 3072 | 128256 | 5.98 | yes (user measured peak ~7.5 GiB) | yes | yes | both | Llama 3.2 Community License; gated |
| Llama-3.2-1B | 1.24B (computed) | 16 | 2048 | 128256 | 2.30 | yes | yes | yes | both | same; gated |
| Gemma-2-2B | 2.6B (computed; HF metadata says "3B") | 26 | 2304 | 256000 | 4.87 | yes | not listed (tagged English) | not listed | both | Gemma license; gated |
| Gemma-3-1B | ~1.0B (computed) | 26 | 1152 | 262144 | 1.86 | yes | "140+ languages" training claim | same | pt and it; text-only (Gemma3ForCausalLM) | Gemma license; gated |
| Gemma-3-4B | 4B | 34 | 2560 | 262208 | text-only 7.23; with SigLIP vision tower larger | no (4-bit needed) | "140+ languages" | same | pt and it; multimodal (Gemma3ForConditionalGeneration) | Gemma license; gated |
| Gemma-4 E2B | 2.3B effective, 5.1B with embeddings | 35 | 1536 | 262144 (+262144 per-layer-input vocab, 256-dim per-layer embeddings) | 5.1B x 2 = about 9.5 (computed) | no in bf16 | "140+ languages pretrain, 35+ out-of-box"; Hindi/Spanish not named | same | multimodal (text+image+audio) | Apache 2.0; tokenizer downloaded without login |
| Gemma-4 E4B | 4.5B effective, 8B with embeddings | 42 | not fetched | 262K | about 15 (computed from 8B) | no | same | same | multimodal | Apache 2.0 |
| Qwen2.5-1.5B-Instruct | 1.54B (computed) | 28 | 1536 | 151936 | 2.88 | yes | not in the listed 29-language claim | named | base and instruct | Apache 2.0 (1.5B, UNVERIFIED for that size); 3B uses qwen-research license |
| Qwen2.5-3B-Instruct | 3.09B (2.77B non-emb) | 36 | 2048 | 151936 | 5.75 | yes (tight) | not mentioned | named | base and instruct | qwen-research (non-commercial-style); not gated |
| Qwen3-1.7B | 1.7B (1.4B non-emb per card, UNVERIFIED for the non-emb figure) | 28 | 2048 | 151936 | 3.20 | yes | yes (119 languages; Hindi appears in Qwen3 eval language lists) | yes | Base and post-trained (hybrid thinking) | Apache 2.0 |
| Qwen3-4B | 4.0B (3.6B non-emb) | 36 | 2560 | 151936 | 7.49 | no (4-bit) | yes (119 langs) | yes | Base and post-trained | Apache 2.0, not gated |
| Qwen3.5-2B | "2B" LM + vision encoder | 24 (18 linear-attention DeltaNet, 6 full attention) | 2048 | 248320 | LM about 3.7 if 2B (computed from the card's "2B", UNVERIFIED exact) | likely yes | "201 languages and dialects"; Hindi/Spanish not named | same | Base and post-trained | Apache 2.0; config fetched without login |
| Qwen3.5-4B | 4B LM; safetensors metadata 5B incl. vision | 32 (8 x [3 DeltaNet + 1 gated attention]) | not fetched | 248320 | LM about 7.5 (computed) | no (4-bit) | 201 languages (not named) | same | post-trained (base UNVERIFIED) | Apache 2.0 |
| Phi-4-mini-instruct | 3.8B (HF says 4B) | 32 | 3072 | 200064 | 7.14 | no (4-bit) | NOT listed (23 languages listed; Spanish yes) | yes | instruct only on this card | MIT |
| SmolLM3-3B | 3B | 36 | 2048 | 128256 | 5.73 | yes | NOT listed (6 native: EN, FR, ES, DE, IT, PT; AR/ZH/RU with fewer tokens) | yes | base and instruct | Apache 2.0 |
| Granite-3.3-2B-base | 2.5B | 40 | 2048 | 49152 | 4.72 | yes | NOT listed (12 languages) | yes | base and instruct | Apache 2.0 |
| Sarvam-1 | card says "2B"; config computes to 2.52B (safetensors metadata says "3B") | 28 | 2048 | 68096 (untied) | 4.70 | yes | yes (10 Indic languages + English) | NOT listed | base (text completion) | Sarvam non-commercial license; gating not stated on card |
| BLOOMZ-1b7 | 1.7B | 24 | 2048 | 250880 | about 3.2 (computed: 1.7B x 2 B) | yes | card page does not list languages (points to BLOOM/xP3 pages) | same | instruct-tuned (xP3 mixture) | bigscience-bloom-rail-1.0 |
| Tiny Aya (base/global/earth/fire/water) | 3.35B (0.5B embedding, 2.8B non-embedding) | 36 | 2048 | about 262K | 6.24 (computed from 3.35B) | borderline (about 0.3 GiB more than Llama-3.2-3B) | yes (70+ languages; "fire" variant targets South Asian languages) | yes per HF card | base plus 4 chat variants | CC-BY-NC 4.0 (non-commercial); gated |

Sources and notes by row:
- Llama-3.2-3B / 1B: config values from the [unsloth mirror configs](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/raw/main/config.json) and [1B config](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/raw/main/config.json) (the official meta-llama configs returned HTTP 401, gated). Official-languages sentence: "English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported" — [Llama-3.2-1B-Instruct card (unsloth mirror)](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct).
- Gemma-2-2B: config from [unsloth/gemma-2-2b](https://huggingface.co/unsloth/gemma-2-2b/raw/main/config.json); the same card page tags language as English only and lists "License: gemma" — [card](https://huggingface.co/unsloth/gemma-2-2b). Official google/gemma-2-2b config returned 401 (gated).
- Gemma-3: 4B config shows Gemma3ForConditionalGeneration with text_config hidden 2560, 34 layers, vocab 262208 and a SigLIP vision tower (27 layers, hidden 1152) — [config](https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json). 1B config is Gemma3ForCausalLM, hidden 1152, 26 layers, vocab 262144 — [config](https://huggingface.co/unsloth/gemma-3-1b-pt/raw/main/config.json). Card: "over 140 languages", repo gated, license tag "gemma" — [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt). Google's model page says context is 128K for 4B/12B/27B and 32K for 1B/270M and states Gemma 3 models are multimodal and trained on 140+ languages; it does not state the 1B is English-only — [model card](https://ai.google.dev/gemma/docs/core/model_card_3).
- Gemma-4: E2B card says 2.3B effective, 5.1B with embeddings, 35 layers, 262K vocab, per-layer embeddings, 35+ languages out-of-box, 140+ pretraining, Apache 2.0 — [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B); config shows Gemma4ForConditionalGeneration, hidden 1536, 35 layers, hidden_size_per_layer_input 256, 8 heads, 1 KV head, intermediate 6144, 20 KV-shared layers, audio and vision configs present — [config](https://huggingface.co/google/gemma-4-E2B/raw/main/config.json). E4B: 4.5B effective, 8B with embeddings, 42 layers — [card](https://huggingface.co/google/gemma-4-E4B). The example code on the E2B card loads it with AutoModelForMultimodalLM; text-only loading is not documented there.
- Qwen2.5: configs for [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json) (hidden 2048, 36 layers, vocab 151936, 16 heads, 2 KV heads, tied) and [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json) (hidden 1536, 28 layers). Card: 3.09B params, 2.77B non-embedding, "over 29 languages" naming Spanish but not Hindi, license qwen-research — [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- Qwen3: configs [1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) (hidden 2048, 28 layers, 16 heads, 8 KV heads) and [4B](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json) (hidden 2560, 36 layers, 32 heads, 8 KV heads). Qwen3-4B-Base card: 4.0B (3.6B non-embedding), 36 layers, 36T tokens across 119 languages, Apache-2.0 — [card](https://huggingface.co/Qwen/Qwen3-4B-Base). Qwen3 report: 119 languages (29 in Qwen2.5), vocabulary 151,669; Table 10 lists Hindi in the Multi-IF, INCLUDE and MMMLU language sets — [arXiv 2505.09388](https://arxiv.org/html/2505.09388).
- Qwen3.5: 2B config is Qwen3_5ForConditionalGeneration, text hidden 2048, 24 layers (every 4th is full attention, other 18 linear_attention), vocab 248320, tied, vision tower depth 24 — [config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json); card: Apache-2.0, 201 languages and dialects, Qwen3.5-2B-Base exists — [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). 4B card: 32 layers, hybrid Gated DeltaNet + gated attention, FFN 9216, vocab 248,320, safetensors 5B incl. vision — [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- Phi-4-mini: config hidden 3072, 32 layers, vocab 200064, 24 heads, 8 KV heads, Phi3ForCausalLM, tied — [config](https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json). Card lists 23 languages (Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian); Hindi is not among them; MIT — [card](https://huggingface.co/microsoft/Phi-4-mini-instruct). Phi-3.5-mini was not separately fetched (3.8B class, similar size, so it would not fit bf16 either; UNVERIFIED).
- SmolLM3: config hidden 2048, 36 layers, vocab 128256, 16 heads, 4 KV heads, tied — [config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/config.json); card: 6 natively supported languages (English, French, Spanish, German, Italian, Portuguese), Arabic/Chinese/Russian with fewer tokens, Apache 2.0, SmolLM3-3B-Base available — [card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).
- Granite-3.3-2B: config hidden 2048, 40 layers, vocab 49152, GraniteForCausalLM — [config](https://huggingface.co/ibm-granite/granite-3.3-2b-base/raw/main/config.json); card lists 12 languages (English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, Chinese), Apache 2.0 — [card](https://huggingface.co/ibm-granite/granite-3.3-2b-base).
- Sarvam-1: LlamaForCausalLM, hidden 2048, 28 layers, vocab 68096, 16 heads, 8 KV heads, intermediate 11008, untied — [config](https://huggingface.co/sarvamai/sarvam-1/raw/main/config.json); card: 10 Indic languages plus English, base model, non-commercial license — [card](https://huggingface.co/sarvamai/sarvam-1).
- BLOOMZ-1b7: config n_embed 2048, n_layer 24, vocab 250880, 16 heads — [config](https://huggingface.co/bigscience/bloomz-1b7/raw/main/config.json); license bigscience-bloom-rail-1.0 — [card](https://huggingface.co/bigscience/bloomz-1b7).
- Tiny Aya: 3.35B, 70+ languages including Hindi and Spanish, CC-BY-NC 4.0, gated, sliding-window layers (window 4096) with every fourth layer global attention, 8K context — [tiny-aya-base card](https://huggingface.co/CohereLabs/tiny-aya-base); paper Table 2: 36 layers, embedding dim 2048, FFN 11008, 16 heads, 4 KV heads, 3.35B total (0.5B embedding, 2.8B non-embedding), vocabulary 262k — [arXiv 2603.11510](https://arxiv.org/html/2603.11510). The tokenizer files were not downloadable without login (my fetch failed), so I could not measure its fertility.

### Inferences
- Computed bf16 weights: the 2B-class models leave 3 to 4 GiB headroom; Qwen3-4B (7.49), Gemma-3-4B text-only (7.23) and Phi-4-mini (7.14) leave under 1 GiB of the 7.9 GiB usable, so they cannot be run with autograd/editing in bf16 in this setup. Large vocabularies matter less than depth for weights (embedding table is 0.5 to 1.25 GiB for the 250K-vocab models), but a 250K-vocab model produces 2x the logits memory of Llama-3 if full-sequence logits are materialised.
- Of the not-yet-planned models, Qwen3.5-2B and Tiny Aya are the only ones that combine explicit multilingual breadth with a size close to the Llama-3.2-3B budget. Tiny Aya (6.24 GiB weights) is roughly 0.26 GiB heavier than Llama-3.2-3B, so it is borderline against the user's 7.5 GiB measured peak.
- Architectural differences from Llama: Qwen2.5/Qwen3 are Llama-like (GQA, SwiGLU; Qwen3 adds QK-norm); Gemma-2/3 add sandwich norms, soft-capping (Gemma-2), sliding-window attention, head_dim 256; Qwen3.5 is a hybrid with Gated DeltaNet layers (only 6 of 24 layers are softmax attention) and is the most different; Tiny Aya uses sliding-window plus global NoPE layers. Sarvam-1 is literally LlamaForCausalLM, so it adds no architectural diversity.

### Gaps
- Official meta-llama and google Gemma-2/3 repos returned 401 for config.json; values come from unsloth mirrors, not the originals. Mirrors are normally byte-identical copies, but I did not diff them.
- Hidden size for Qwen3.5-4B and Gemma-4 E4B, exact LM-only parameter count for Qwen3.5-2B, and whether Qwen3.5-4B has a Base release were not confirmed.
- Whether Gemma-3-4B or Qwen3.5 text-only weights can be loaded without the vision tower in the current transformers version (transformers 5.18.0 is installed locally) was not tested.
- Aya Expanse / Aya-23 small variants: only 8B and larger exist in my knowledge; not verified here. OpenHathi (Llama-2-7B based) and Airavata (7B) are 7B-class, out of budget; not fetched. EXAONE small models, mGPT (1.3B), BLOOM-3B, Mistral-small-class (Ministral 3B) were not fetched; they are skipped for lack of verified data, not shown to be bad.

---

## 2. Hindi tokenization cost (tokenizer fertility) and byte-splitting

### Takeaway
Own measurement on parallel XQuAD paragraphs shows a sharp split: Gemma-3/4 (1.49 tokens per Hindi word), Sarvam-1 (1.51), BLOOM (1.43) and Phi-4-mini (1.76) are efficient; Gemma-2 (2.12), Qwen3.5 (2.32) and Llama-3 / SmolLM3 (2.80) are acceptable; Qwen2.5 and Qwen3 (5.00, with 53% of Hindi tokens being partial UTF-8 byte fragments), Granite-3.3 (6.18, 57% byte fragments) and Mistral-7B-v0.3 (5.56) are poor. The Qwen2.5/Qwen3 byte-splitting directly harms first-token object metrics in Hindi.

### Cited Findings

Method (own measurement, reproducible): tokenizers downloaded from the HF repos listed below; text = the 240 paragraphs (contexts) of XQuAD in English, Spanish and Hindi, which are professional translations of the same SQuAD paragraphs ([google/xquad](https://huggingface.co/datasets/google/xquad)); words = whitespace-split; tokens counted without special tokens; "byte-fragment tokens" = Hindi tokens that decode on their own to a string containing U+FFFD (i.e. a partial UTF-8 sequence). Word counts: en 29,724, es 34,363, hi 33,850.

| Tokenizer (repo used) | Vocab | EN tok/word | ES tok/word | HI tok/word | HI chars/token | HI byte-fragment token share | Tokens for Hindi city/entity names (Delhi, Mumbai, India, France, Paris, Mahatma Gandhi, Ganga river, Rabindranath Tagore, Spain, Madrid) |
|---|---|---|---|---|---|---|---|
| BLOOM / BLOOMZ ([bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)) | 250,680 | 1.30 | 1.29 | 1.43 | 3.78 | 0.0% | 1,1,1,3,3,2,2,5,2,3 |
| Gemma-3 ([gemma-3-1b-pt](https://huggingface.co/unsloth/gemma-3-1b-pt)) | 262,145 | 1.33 | 1.40 | 1.49 | 3.63 | 0.0% | 1,1,1,3,2,3,2,5,2,3 |
| Gemma-4 ([gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)) | 262,144 | 1.33 | 1.40 | 1.49 | 3.63 | 0.0% | identical to Gemma-3 |
| Sarvam-1 ([sarvam-1](https://huggingface.co/sarvamai/sarvam-1)) | 68,096 | 1.57 | 2.56 | 1.51 | 3.60 | 1.4% | 1,1,1,3,1,3,2,4,1,3 |
| Phi-4-mini ([Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)) | 200,029 | 1.30 | 1.39 | 1.76 | 3.08 | 0.4% | 2,2,1,3,3,4,3,6,2,4 |
| Gemma-2 ([gemma-2-2b](https://huggingface.co/unsloth/gemma-2-2b)) | 256,000 | 1.33 | 1.38 | 2.12 | 2.55 | 0.0% | 2,3,2,3,3,6,4,9,2,6 |
| Qwen3.5 ([Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)) | 248,077 | 1.35 | 1.45 | 2.32 | 2.34 | 0.1% | 3,3,3,4,3,7,4,9,3,6 |
| Llama-3.2 ([Llama-3.2-3B-Instruct, unsloth](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)) | 128,256 | 1.31 | 1.64 | 2.80 | 1.93 | 0.4% | 4,4,2,5,3,8,5,10,3,5 |
| SmolLM3 ([SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)) | 128,256 | 1.31 | 1.64 | 2.80 | 1.93 | 0.4% | identical to Llama-3 (same tokenizer) |
| Qwen2.5 ([Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)) | 151,665 | 1.36 | 1.67 | 5.00 | 1.08 | 52.9% | 6,5,4,8,5,13,8,16,5,7 |
| Qwen3 ([Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)) | 151,669 | 1.36 | 1.67 | 5.00 | 1.08 | 52.9% | identical to Qwen2.5 |
| Mistral-7B-v0.3 ([unsloth mirror](https://huggingface.co/unsloth/mistral-7b-v0.3)) | 32,768 | 1.49 | 1.86 | 5.56 | 0.98 | 4.8% | 7,6,5,8,6,14,9,20,6,8 |
| Granite-3.3 ([granite-3.3-2b-base](https://huggingface.co/ibm-granite/granite-3.3-2b-base)) | 49,152 | 1.63 | 2.03 | 6.18 | 0.88 | 56.7% | 7,8,5,10,6,16,12,20,6,10 |
| Tiny Aya | about 262K | not measured | not measured | not measured | | | tokenizer gated |

Published numbers (secondary or vendor-reported, different corpora, so do not mix with the table above):
- Sarvam says other multilingual models often need "4 to 8 tokens per word compared to 1.4 for English", and Sarvam-1's tokenizer has fertility "1.4-2.1 across all supported languages" (no Hindi-specific number) — [Sarvam-1 blog](https://sarvam.ai/blogs/sarvam-1); same claim on the [HF card](https://huggingface.co/sarvamai/sarvam-1).
- A community project reports Gemma-3 about 3.38 and Llama-3 about 8.55 fertility on its own corpus (unclear unit; community source, unverified) — [indictok-256k](https://huggingface.co/vivekvar/indictok-256k). These numbers disagree in scale with mine (Llama-3 2.8 tokens/word on XQuAD) so treat as unverified. The Gnani.ai table (April 2026) reports Gemma 3 27B about 2.11 and Qwen 3.5 about 3.29 averages across its own languages, columns unlabeled in the excerpt I saw — [Gnani.ai](https://gnani.ai/resources/research/indic-tokenizer-nemotron).
- Tiny Aya paper: Figure 3 compares tokens per character by script against Gemma3-4B, Qwen3-4B and SmolLM3-3B, and the text highlights Khmer, Telugu, Gujarati, Lao, Ge'ez rather than Hindi; the figure itself could not be read — [arXiv 2603.11510](https://arxiv.org/html/2603.11510).

### Inferences
- A tokenizer that frequently emits partial-UTF-8 tokens (Qwen2.5, Qwen3, Granite, to a lesser extent Mistral) means the first token of a Hindi object is often a meaningless byte fragment shared by many different objects. First-token-probability metrics on Hindi would then be near uninformative, and any metric must use full-object log-probability or multi-token exact match. Gemma-3/4, Sarvam-1, BLOOM, Phi-4-mini, Gemma-2 and Qwen3.5 avoid this.
- Qwen3.5 changed the Qwen tokenizer from 151K to 248K vocabulary and Hindi fell from 5.0 to 2.32 tokens/word with almost no byte fragments (0.1%), so the "Qwen is bad for Hindi" conclusion applies to Qwen2.5/Qwen3, not Qwen3.5.
- Tokenizer sharing is useful: SmolLM3 shares Llama-3's tokenizer exactly, and Gemma-4 shares Gemma-3's (same fertility on this corpus, though I did not diff vocab files). Sarvam-1 is the only model here with Hindi better than English-like fertility but is worse than all others on Spanish (2.56).
- Spanish: all large-vocabulary tokenizers are at 1.29 to 1.67; only Sarvam-1 (2.56) and Granite (2.03) and Mistral (1.86) are noticeably worse.

### Gaps
- Tiny Aya fertility could not be measured (gated tokenizer) and its paper's Hindi bar is in a figure.
- My corpus is Wikipedia-style translated text (XQuAD); fertility on short entity strings (objects in fact triples) differs and is only sampled by ten strings above. Whitespace-word counts are language dependent, so cross-language ratios should be read with care.
- Phi-3.5-mini, mGPT, Aya-23, EXAONE and Ministral tokenizers were not measured.

---

## 3. Prior knowledge-editing / causal-tracing / multilingual factual-recall papers using these models (arXiv IDs verified by opening the page)

### Takeaway
Llama-3.2-3B and Gemma-2-2B are the only candidate small models I found in multilingual factual-recall/editing papers; Qwen2.5-7B, BLOOM-7B and Qwen-7B appear at 7B scale. I found no knowledge-editing or causal-tracing paper using Gemma-3, Gemma-4, Qwen3, Qwen3.5, Phi-4-mini, SmolLM3, Sarvam-1 or Tiny Aya (limited search, not exhaustive).

### Cited Findings
- arXiv 2502.17955, "Language Models' Factuality Depends on the Language of Inquiry" (X-FaKT): evaluates 14 models including Llama-3.2-3B, Gemma-2-2B, Llama-3.2-1B, Phi-3.5-4B (as listed in its Table 2), plus Gemma-2-9B/27B, Llama-3-8B/70B, Phi-3/Phi-4-14B, Mistral-7B-v0.2, DeepSeek-7b, Orca-2-7B. 13 languages: English, Chinese, French, Japanese (high); Hindi, Russian, Arabic, Greek (medium); Nepali, Ukrainian, Turkish, Swahili, Thai (low). Hindi included, Spanish not — [arXiv HTML](https://arxiv.org/html/2502.17955); title and topic confirmed by search result [arXiv PDF](https://arxiv.org/pdf/2502.17955v1).
- arXiv 2504.20022, "Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages" (Rohera, Ginimav, Sawant, Joshi): GPT-4o, Gemma-2-9B, Gemma-2-2B, Llama-3.1-8B; English plus 19 Indic languages (IndicQuest); models do better in English and hallucinate more in low-resource Indic languages — [arXiv](https://arxiv.org/abs/2504.20022).
- arXiv 2406.17764, BMIKE-53 ("Investigating Cross-Lingual Knowledge Editing with In-Context Learning", ACL 2025): two models, Llama3.2-3B and Llama3.1-8B; 53 languages including Spanish (es) and Hindi (hi) (Table 8); finds non-Latin-script languages do worse — [arXiv HTML v2](https://arxiv.org/html/2406.17764v2), summary also at [alphaxiv/search snippet](https://alphaxiv.org/benchmarks/university-of-oxford/bmike-53).
- arXiv 2506.10800, "Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints" (LangEdit): GPT-J-6B, Llama3-8B, Qwen2.5-7B; six languages English, German, Dutch, Spanish, French, Chinese; no Hindi — [arXiv HTML](https://arxiv.org/html/2506.10800), abstract at [arXiv](https://arxiv.org/abs/2506.10800).
- arXiv 2408.10646, "Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs" (Ifergan et al.): repurposes knowledge editing to measure sharing; six 7B models (BLOOM-7B, Qwen-7B, Llama-2-7B, Mistral-7B-v0.1, Chinese-llama-2-7B, Hebrew-Mistral-7B); 13 languages including Spanish and Hindi; finds script similarity is a major factor in sharing — [arXiv HTML](https://arxiv.org/html/2408.10646), abstract at [arXiv](https://arxiv.org/abs/2408.10646).
- arXiv 2410.14387, "How Do Multilingual Language Models Remember Facts?" (Fierro, Foroutan, Elliott, Søgaard): causal-patching analysis on XGLM-7.5B, EuroLLM-9B, mT5-xl; languages include Spanish but the verified list does not include Hindi; subject enrichment is language-independent while object extraction is language-dependent — [arXiv abs](https://arxiv.org/abs/2410.14387), model/language details from [arXiv HTML v3](https://arxiv.org/html/2410.14387v3).
- arXiv 2402.10588, "Do Llamas Work in English? On the Latent Language of Multilingual Transformers" (Wendler et al., ACL 2024): Llama-2 family, logit-lens style analysis showing a concept space closer to English — [arXiv PDF](https://arxiv.org/pdf/2402.10588); I did not open the paper's language list, only the search summary.
- arXiv 2505.14393, "Editing Across Languages: A Survey of Multilingual Knowledge Editing" (Durrani, Mousi, Dalvi): organizes methods into parameter-based, memory-based, fine-tuning and hypernetwork; names language anisotropy as an obstacle; names no specific models on the abstract page — [arXiv](https://arxiv.org/abs/2505.14393).
- arXiv 2603.11510 (Tiny Aya) reports cross-language knowledge averages only (Table 4, Global MMLU 42 languages, INCLUDE 44, Global PIQA 116); no editing use — [arXiv HTML](https://arxiv.org/html/2603.11510).

### Inferences
- For comparability with the literature, Llama-3.2-3B, Llama-3.2-1B and Gemma-2-2B are the best-precedented small models (X-FaKT, BMIKE-53 and the Indic factuality paper). Adding Qwen would link to LangEdit (Qwen2.5-7B), but only at 7B scale and not with Hindi.
- Hindi plus Spanish together appear in only two found papers (2406.17764 and 2408.10646), and neither uses a sub-4B model other than Llama-3.2-3B.

### Gaps
- My searches were not exhaustive (about 8 queries). Papers such as MzsRE/MLaKE/BMIKE variants beyond those named, or CounterFact-style Hindi datasets, were not systematically surveyed. No verified prior use of Qwen3, Qwen3.5, Gemma-3, Gemma-4, Phi-4-mini, SmolLM3, Sarvam-1 or Tiny Aya in editing/tracing was found; absence of evidence only.
- 2410.14387 and 2402.10588 per-language lists were only partly confirmed.

---

## 4. Evidence of factual-recall quality in Hindi and Spanish

### Takeaway
Direct per-language Hindi and Spanish knowledge numbers for sub-4B models are scarce. The best primary data: Meta's multilingual MMLU for Llama-3.2 (Hindi 43.3 vs Spanish 55.1 at 3B; 33.5 vs 41.5 at 1B), Sarvam's translated Indic benchmarks (Gemma-2-2B is near chance on Indic MMLU), Gemma-3 PT Global-MMLU-Lite (1B 24.9 about chance; 4B 57.0), and Qwen3/Qwen3.5 aggregate multilingual scores. SmolLM3's Spanish Global-MMLU (CF) is only 38.5.

### Cited Findings
- Llama-3.2 multilingual MMLU (5-shot, from Meta's card table; instruct, bf16 column): Llama-3.2-1B: Spanish 41.5, Hindi 33.5, Portuguese 39.8, Italian 39.8, German 39.2, French 40.5, Thai 34.7; Llama-3.2-3B: Spanish 55.1, Hindi 43.3, Portuguese 54.5, Italian 53.8, German 53.3, French 54.6, Thai 44.5. MGSM (CoT, 0-shot): 1B 24.5, 3B 58.2 (aggregate only) — [meta-llama/Llama-3.2-3B-Instruct card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). The unsloth mirror card lacks this table.
- Sarvam-1 blog (translated Indic benchmarks, Indic average unless noted): Gemma-2-2B MMLU Indic 29.86 (English 45.06), ARC-C Indic 30.45, TriviaQA Indic 40.58 (English 78.13), BoolQ 57.87; Llama-3.2-3B MMLU Indic 33.39 (English 52.69), ARC-C 36.50, TriviaQA Indic 46.00 (English 87.28), BoolQ 64.39; Sarvam-1 MMLU Indic 44.44 (English 47.64), ARC-C 58.50, TriviaQA Indic 90.62 (English 93.38), BoolQ 80.68. IndicGenBench (one-shot): XQuAD F1 Gemma-2-2B 32.49, Llama-3.2-3B 41.69, Sarvam-1 41.58; XORQA F1 12.28 / 18.37 / 25.27 — [Sarvam-1 blog](https://sarvam.ai/blogs/sarvam-1). Vendor-reported; Indic average (not Hindi alone). The blog does not include Qwen-2.5-3B.
- Gemma-3 pretrained, from HF card: Global-MMLU-Lite 1B 24.9, 4B 57.0; MGSM 1B 2.04, 4B 34.7; XQuAD (all) 1B 43.9, 4B 68.0; IndicGenBench 1B 41.4, 4B 57.2; FloRes 29.5 / 39.2; ECLeKTic 4.69 / 11.0 — [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt). Per-language Hindi/Spanish values not given.
- Gemma-4 E2B instruct MMMLU 67.4%; E4B 76.6% — [E2B card](https://huggingface.co/google/gemma-4-E2B), [E4B card](https://huggingface.co/google/gemma-4-E4B).
- Qwen3 report: Qwen3-1.7B-Base MGSM 50.71, MMMLU 63.27, INCLUDE 45.57; Qwen3-4B-Base MGSM 67.74, MMMLU 71.42, INCLUDE 56.29; post-trained Qwen3-1.7B thinking: MMMLU 59.1, INCLUDE 51.8; non-thinking: MMMLU 48.3, INCLUDE 42.6; Qwen3-4B thinking MMMLU 69.8 / INCLUDE 61.8, non-thinking MMMLU 61.7 / INCLUDE 53.8. Belebele Indo-European average: Qwen3-1.7B 69.7 (thinking) / 58.8 (non-thinking), Qwen3-4B 82.2 / 76.0, Gemma-3-4B-IT 71.8. Per-language Hindi and Spanish rows for small models were not in the text I read — [arXiv 2505.09388](https://arxiv.org/html/2505.09388).
- Qwen3.5-2B (thinking): MMMLU 63.1, MMLU-ProX 52.3 (29-language average), NOVA-63 46.4, INCLUDE 55.4, Global PIQA 69.3; Qwen3.5-0.8B MMMLU 44.3 — [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). Qwen3.5-4B: MMMLU 76.1, MMLU-ProX 71.5, NOVA-63 54.3, INCLUDE 71.0, Global PIQA 78.9 — [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- Phi-4-mini card multilingual MMLU (5-shot) / MGSM: Phi-4-mini 49.3 / 63.9; Phi-3.5-mini 51.8 / 49.6; Llama-3.2-3B-Ins 48.1 / 44.6; Qwen2.5-3B-Ins 55.9 / 53.5 (language set not specified on the card) — [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct).
- SmolLM3-3B Base: Spanish Belebele 47.00, Global MMLU (CF) 38.51, MLMM HellaSwag 65.85, Flores-200 48.25; Hindi not evaluated — [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).
- Tiny Aya Global: Global MMLU (42 languages) 44.9±7.3, INCLUDE (44) 45.1±11.1, Global PIQA (116) 68.3±10.6; vs Gemma3-4B 45.3 / 48.9 / 70.8, Qwen3-4B 49.3 / 52.2 / 74.6, Qwen3.5-4B 54.8 / 56.9 / 79.31; averages only, no Hindi/Spanish split — [arXiv 2603.11510](https://arxiv.org/html/2603.11510). The HF card gives no numeric scores and says CoT math such as MGSM is comparatively weaker — [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base).
- X-FaKT (2502.17955) and the Indic factuality paper (2504.20022) both report better accuracy in English than in Hindi/Indic languages across sizes; per-model Hindi numbers for Gemma-2-2B and Llama-3.2-3B were not extracted — see section 3 links.
- Qwen2.5-3B: the card claims 29 languages without Hindi; no Hindi-specific score found.

### Inferences
- Llama-3.2-3B (Hindi MMLU 43.3 vs Spanish 55.1) already shows a Hindi gap of roughly 12 points against Spanish; the 1B model (33.5) is close to the 25% chance floor for a 4-choice test, so small-model Hindi factual coverage will be thin. Gemma-3-1B PT at 24.9 on Global-MMLU-Lite is effectively chance and is not a good source of correct multilingual facts.
- Gemma-2-2B is the most worrying planned model for Hindi: Indic MMLU 29.86 is near chance and Indic TriviaQA 40.58 is about half its English value (78.13). Expect few facts answered correctly in all three languages unless the fact set is made of very well-known entities (capitals, countries).
- For breadth in all three languages, the best primary-source indicators point to Gemma-3-4B (but not bf16-fittable), Qwen3.5-2B/4B, Qwen3-4B (poor Hindi tokenization), and Llama-3.2-3B itself. Sarvam-1 is exceptionally strong on Hindi trivia (TriviaQA Indic 90.62 vs 46.00 for Llama-3.2-3B) but has no Spanish support, so its Spanish coverage is unknown and likely weak.
- A reasonable working assumption (UNVERIFIED): reaching n>=50 facts correct in all three languages needs to start from a candidate pool of several hundred high-frequency facts (country/capital/language/continent style) for models in the 1-2B class; this must be measured empirically per model.

### Gaps
- Per-language Hindi and Spanish factual-QA (not MMLU) accuracy was not found for any candidate except via Meta's MMLU table for Llama-3.2.
- No published Hindi-specific knowledge score for Gemma-2-2B, Qwen2.5-3B, Phi-4-mini, SmolLM3, Granite, Qwen3.5, Gemma-3-4B or Tiny Aya was found (only aggregates).
- Qwen3 report per-language Hindi/Spanish rows for 1.7B/4B were not in the portion I could read (Table 36 lists hin_Deva and spa_Latn without scores in the text; Table 37 gives family averages only).

---

## 5. Ranked recommendation: top models to ADD

Goal assumed: at least 50 facts answered correctly in English, Spanish and Hindi; architecture different from Llama; feasible on 8 GB.

### Takeaway
Rank: (1) Qwen3.5-2B (best fit overall, newest), (2) Gemma-3-4B in 4-bit or Tiny Aya-base in bf16 (best knowledge-per-language evidence but each has a feasibility catch), (3) Qwen3-1.7B (safe, fits, but Hindi byte-splitting), (4) Sarvam-1 as a Hindi-strong control (not architecture-diverse). Keep Gemma-2-2B as planned but expect Hindi to be the bottleneck. Skip Phi-4-mini, SmolLM3, Granite, Qwen2.5 and Gemma-3-1B for this goal.

### Cited Findings (ranking evidence is in sections 1 to 4; per-model rationale below)

1. Qwen3.5-2B (post-trained or Base). Reasons: Hindi 2.32 tokens/word with 0.1% byte fragments (own measurement, section 2); explicit "201 languages" claim and MMMLU 63.1, INCLUDE 55.4 for the 2B ([card](https://huggingface.co/Qwen/Qwen3.5-2B)); 24 layers, d_model 2048, tied embeddings, about 3.7 GiB LM weights (computed); Apache 2.0, ungated; architecturally the most different from Llama (hybrid Gated DeltaNet: 18 of 24 layers are linear attention). Risks: only 6 full-attention layers, so classical attention-based tracing differs; MLP layers are standard FFN so ROME/MEMIT-style MLP editing is still defined, but hook points differ (UNVERIFIED for the tooling); multimodal wrapper class (Qwen3_5ForConditionalGeneration), thinking-mode defaults (prefer the Base model for factual completion); fast kernels for DeltaNet may not install on Windows (UNVERIFIED); no prior editing papers found.
2. Gemma-3-4B (pt) in 4-bit, or Tiny Aya-base in bf16.
   - Gemma-3-4B: Hindi 1.49 tokens/word with 0% byte fragments, Global-MMLU-Lite PT 57.0, IndicGenBench 57.2, XQuAD 68.0 ([card](https://huggingface.co/google/gemma-3-4b-pt)); sliding-window plus global attention, head_dim 256, 34 layers, d_model 2560. Risks: 7.23 GiB text-only bf16 weights (computed) means no autograd in bf16; 4-bit editing is awkward; gated; multimodal checkpoint class (text tower must be loaded alone, UNVERIFIED).
   - Tiny Aya-base: 3.35B, 36 layers, d_model 2048, 262K vocab, explicit Hindi and Spanish among 70+ languages, Global MMLU average 44.9, close to Gemma3-4B 45.3 ([paper](https://arxiv.org/html/2603.11510)); bf16 6.24 GiB (computed). Risks: borderline memory (about 0.26 GiB over Llama-3.2-3B whose peak is 7.5 GiB), gated and CC-BY-NC, tokenizer fertility unmeasured, no prior editing use, architecture details beyond the card (sliding-window plus global NoPE layers) not verified against the HF modeling code.
3. Qwen3-1.7B (Base). Reasons: 3.20 GiB bf16, Apache 2.0, ungated; MMMLU 63.27 and INCLUDE 45.57 for the base ([report](https://arxiv.org/html/2505.09388)); Hindi in its evaluation sets; QK-norm GQA architecture; room to do heavy autograd. Risks: Hindi 5.0 tokens/word and 52.9% byte-fragment tokens (own measurement), so first-token metrics fail and Hindi objects are long token sequences; Qwen3-1.7B knowledge capacity is limited (n>=50 shared facts may be hard).
4. Sarvam-1. Reasons: best Hindi tokenization among small models (1.51) and Indic TriviaQA 90.62 vs Llama-3.2-3B 46.00 ([blog](https://sarvam.ai/blogs/sarvam-1)); 4.70 GiB; Llama architecture is a plus for a controlled "same architecture, different training" comparison. Risks: Spanish unsupported and tokenized at 2.56 tokens/word (own measurement), so shared-fact coverage in Spanish is uncertain; LlamaForCausalLM adds no architecture diversity; non-commercial license; base model only.
5. (Optional) Llama-3.2-1B as a within-family scale control (Hindi MMLU 33.5, near chance), and Gemma-2-2B (already planned).

Not recommended for this goal: Phi-4-mini (Hindi not supported, 7.14 GiB bf16 so 4-bit only), Qwen3-4B / Qwen3.5-4B (7.5 GiB bf16; Qwen3-4B also has byte-split Hindi), SmolLM3-3B (no Hindi; same tokenizer as Llama-3), Granite-3.3-2B (no Hindi; 6.18 tokens/word, 57% byte fragments), Qwen2.5-1.5B/3B (no Hindi in listed languages; 5.0 tokens/word), Gemma-3-1B (Global-MMLU-Lite 24.9, about chance), Gemma-4 E2B/E4B (multimodal, 5.1B / 8B parameters including per-layer embeddings, about 9.5 GiB bf16 total for E2B; extreme load for 8 GB unless the per-layer embedding tables are offloaded, UNVERIFIED), BLOOMZ-1b7 (good tokenization, Hindi and Spanish are in BLOOM's training languages per the BLOOM project, but I did not verify that on a primary page; instruction tuning on xP3 and weak knowledge vs 2024+ models, UNVERIFIED).

### Inferences
- For a study about the language of factual recall, the Hindi tokenization/byte-splitting factor (Qwen2.5/Qwen3, Granite) can confound the language effect. If Qwen3 is used, the study should not use first-token metrics for Hindi.
- Choosing Qwen3.5-2B plus one of Gemma-3-4B (4-bit) or Tiny Aya yields three different design lineages (hybrid linear attention, Gemma-style local/global attention, Cohere sliding-window) beside the Llama and Gemma-2 baselines.
- Memory plan (computed, not measured): Qwen3.5-2B and Qwen3-1.7B leave more than 4 GiB headroom; Tiny Aya leaves about 1.7 GiB (less than the Llama-3.2-3B case implies, because Llama-3.2-3B peaked at 7.5 GiB with 5.98 GiB weights, so Tiny Aya's peak may reach about 7.8 GiB, near the 7.9 GiB limit). A pilot run of peak memory is needed before committing.

### Gaps
- None of the recommended models was actually loaded or run on the user's GPU; all memory figures are arithmetic, and activation/gradient overhead for the hybrid and sliding-window architectures is unmeasured.
- Counts of facts answerable correctly in all three languages (the n>=50 requirement) can only be determined with a pilot on the actual fact set.
- Whether Gemma-3-4B / Qwen3.5 text towers load without vision weights in the installed transformers 5.18.0 is untested.
- License terms for Gemma-3 and Tiny Aya research use were read only at card level.
