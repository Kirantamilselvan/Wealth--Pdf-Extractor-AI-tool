# Model notes

Keep this file current — it's the source of truth for what's actually true
about the models when you talk about this project in an interview.

| Model      | Checkpoint used              | Status                          |
|------------|-------------------------------|----------------------------------|
| Summarizer | `facebook/bart-large-cnn`     | Off-the-shelf, zero-shot (as of initial build) |
| Risk tagger| `ProsusAI/finbert`             | Off-the-shelf, zero-shot        |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Off-the-shelf |

If you fine-tune any of these on financial filing data, update this table with:
- what data you fine-tuned on (size, source)
- training setup (epochs, hardware, time)
- before/after eval numbers from `eval/results/eval_report.json`

This keeps your resume framing accurate — "evaluated open-source models" is
true today; "fine-tuned open-source models" only becomes true once this table
says so.
