# EXPERIMENT LOG
| run ID | hypothesis | config | result | verdict |
|---|---|---|---|---|
| P0-smoke-gpt2 | model loads and predicts Paris | gpt2 bf16 cuda, prompt "The Eiffel Tower is located in the city of" | top1 Paris p=0.0706, VRAM 0.39 GiB | pipeline OK; not the target model |
