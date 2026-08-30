# Gyeongsang Gold Review Workflow

The training source of truth is human-reviewed dialogue, not automatically
accepted model output. Work in batches of 100 and evaluate after every batch.

## Review file

Prepare an Excel-compatible CSV from a JSONL candidate file:

```bash
python gold_review.py prepare \
  --input data/candidates/gyeongsang_gold_candidates.jsonl \
  --output data/review/gyeongsang_gold_batch_001.csv \
  --limit 100
```

Each candidate record uses this schema:

```json
{"id":"daily-0001","topic":"daily","user":"오늘 뭐 먹으면 좋을까요?","standard_answer":"따뜻한 국물 요리가 좋겠습니다.","candidates":["따뜻한 국물 요리가 좋겠습니더.","따뜻한 국물 요리 드셔 보이소."]}
```

Open the CSV in Excel. Do not edit `id`, `user`, or candidate columns. Fill
`decision` with exactly one of:

- `accept_1`, `accept_2`, or `accept_3`
- `edit`, with the final answer in `edited_answer`
- `reject`

Save it as CSV UTF-8, then validate it:

```bash
python gold_review.py validate \
  --input data/review/gyeongsang_gold_batch_001.csv \
  --require-complete
```

Export only reviewed answers. The command also creates preference pairs from
the unselected candidates for later DPO training:

```bash
python gold_review.py export \
  --input data/review/gyeongsang_gold_batch_001.csv \
  --output-dir data/processed/gyeongsang/gold_v1
```

Outputs:

- `train.jsonl`: reviewed SFT training conversations
- `validation.jsonl`: deterministic held-out conversations
- `preferences.jsonl`: chosen/rejected pairs for DPO
- `summary.json`: exact review and export counts

Do not start a full training run until the first 100 reviewed items pass a
held-out qualitative evaluation. Never merge an automatically generated answer
into the gold set without `accept_*` or `edit` from a native reviewer.
