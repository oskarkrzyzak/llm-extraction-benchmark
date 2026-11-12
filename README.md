## Results

Same 50 questions, same prompts, one run per model (November 12, 2026). Judge: gpt-5.4-mini.

| Model | Extraction | Answers | Avg time per question | Errors |
|---|---|---|---|---|
| gpt-5.4-mini | 46/50 | 47/50 | 1.5 s | 0 |
| claude-haiku-5-5 | 50/50 | 50/50 | 1.8 s | 0 |
| gemini-3.5-flash-lite | 50/50 | 50/50 | 4.0 s | 0 |

Each question is two model calls (extract the items, then answer from the menu data).
Token counts are not compared because every provider counts tokens differently.

## What I learned
- On this test set Claude and Gemini both scored 50/50, so it can't tell them apart. A harder set is needed.
- gpt-5.4-mini returned an empty list for some items that are on the menu, and the failures changed between runs, so it is not stable on this task.
- Using an LLM as a judge helped, but it is not reliable on its own: it passed one wrong answer, and it also tends to favor answers like its own.

## Limitations
- One run per model on 50 questions, so small differences are noise.
- The judge is also one of the tested models (OpenAI).
- Questions needing a filter ("cheapest drink", "anything vegan") are not supported.
- The extraction prompt was tuned on some of these questions, so scores are slightly optimistic.
- Gemini was on the free tier with rate limits, which affects its time.