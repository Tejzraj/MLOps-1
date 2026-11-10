# Phase 2 model-quality risk: stack/queue diagnostic

The verified B0 v1.0.0 service can return high quality categories for reversed
facts and unrelated text. Integration success does not establish grading quality.
The API result now displays:

> Experimental quality prediction, not a factual-correctness check. Results may be incorrect and require human review.

## Exact requests and observed responses

Every request used the same question:

> What is the difference between a stack and a queue?

Every request used this exact reference (single spaces between wrapped lines):

> A stack follows last in, first out (LIFO): push adds an item and pop removes the most recently added item. It is useful for function calls or undo history. A queue follows first in, first out (FIFO): enqueue adds at the back and dequeue removes the oldest item from the front. It is useful for task scheduling or breadth-first search. Both can support O(1) insertion and removal with appropriate implementations.

These are illustrative probes, not dataset rows or newly assigned human labels.
Live POST `/predict` results on 2026-11-10:

| Case | Exact candidate answer | HTTP | Actual predicted_quality |
| --- | --- | --- | --- |
| Correct | A stack uses LIFO and a queue uses FIFO. | 200 | Excellent |
| Incorrect | A stack uses FIFO and a queue uses LIFO. | 200 | Excellent |
| Nonsense | Bananas and airplanes are flying everywhere. | 200 | Good |
| Empty | Empty string (`""`) | 422 | None; candidate_answer validation failure |
| Complete correct | A stack uses last in, first out (LIFO): push adds to the top and pop removes the most recently added item. Stacks are used for function calls and undo history. A queue uses first in, first out (FIFO): enqueue adds at the back and dequeue removes the oldest item from the front. Queues are used for task scheduling and breadth-first search. | 200 | Excellent |
| Reported original | stack is fifo and queue is lifo | 200 | Excellent |

Every successful response reported `model_version: 1.0.0`. Labels are preserved
exactly; no corrections, replacement scores or confidence values were generated.

## Confirmed findings

The checksum-verified frozen pipeline produces 2,824 features: 2,818 candidate
TF-IDF features, five length statistics and one reference cosine similarity.
Question text is unused. The short correct/incorrect answers had **identical
complete feature vectors**: zero differing entries, maximum difference 0.0 and
cosine similarity 0.1889571100473404. Both vectorizers discarded the two
distinguishing analyzed n-grams. This representation cannot distinguish the
reversal. Nonsense had zero nonzero candidate TF-IDF features and zero cosine
similarity; only length statistics remained, yet it received Good.

Class imbalance may contribute but is not the sole established cause. Verified
training counts were Poor 62, Needs Improvement 233, Good 277, Excellent 989;
B0 already uses balanced class weights. Historical reused-test macro-F1 was
0.392525, with Poor recall 0.300000 and Good recall 0.081395
([evaluation report](final_evaluation_and_bundle.md)); no benchmark was rerun.
Exact tree-path causes were not attributed.

No pipeline/mapping mismatch was found: mappings consistently use 0–3 for Poor,
Needs Improvement, Good, Excellent; all stored labels match score thresholds.
Inference cleaning changed zero processed fields across existing partitions.
Pair integrity, dependencies, feature ordering and transformer-source checks
passed. Direct predictions matched API outputs for the three short probes.

## Reproduce without changing the model

With the verified local backend running, from the repository root:

```bash
.venv-api/bin/python -B src/api/check_quality.py
```

All inputs, acceptance criteria and connection settings come from root
`params.yaml`. The script calls the API without loading artifacts or altering
predictions. It logs actual responses and checks that the correct answer
outranks the reversal and nonsense avoids Good/Excellent. These are acceptance
criteria, not new human scores. **Both quality checks currently fail.**

Exit **1** means quality failures; **2** means operational/contract errors;
**0** means the configured checks were satisfied. Do not suppress these failures
or describe them as passing tests. The command is opt-in, outside pytest and
regular CI; unrelated checks remain independent. No tests were weakened or
marked xfail. The prior existing API suite passed 26 tests despite these semantic
failures, illustrating the difference between integration and model quality.

## Limits and possible Phase 3 work

These manually selected probes are not a representative benchmark. Passing them
would not establish factual correctness. Feature inspection covers this pair;
API tests establish consistency, not semantics. Historical test results were
previously inspected; minority supports and external generalization remain weak.

Phase 3 should define a human-reviewed rubric and separate contrastive regression
set covering reversals, negation, irrelevance and paraphrases. Evaluate features
that retain relationships or model answer/reference interactions, minority
performance and unfamiliar-input abstention using held-out evidence. Rebalancing
alone cannot recover distinctions lost by the representation. Do not tune on
these probes and claim unseen performance. No retraining, inference changes,
artifact publication or W&B changes were made.
