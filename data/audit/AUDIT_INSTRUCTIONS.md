# Native-speaker check of the facts (362 rows)

Each row is one fact written in English, Spanish and Hindi. Please check:
1. **EN_fact_true?** Is the English fact true? (e.g. "The capital of France is Paris".)
2. **ES_ok? / HI_ok?** Are the Spanish and Hindi subject, answer and prompt correct, natural and the same fact as the English one? Say N if a word is wrong, an alternative common spelling should also be accepted, or the sentence is unnatural or ambiguous.
3. If N, write the correction in the **correction** column (say which language) and any other note in **comment**.

The prompt is an unfinished sentence: the model must continue it with the answer. A prompt may be short or unusual; only mark N if the meaning is wrong or a native speaker would not write it.
Return the file as CSV. Rows are not changed in place; corrections are applied by the researcher and the affected analyses rerun.
Time: about 6 hours for Spanish plus Hindi, for two checkers in parallel.
