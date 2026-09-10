# Semantic Vector Guardrail Algorithm

## Purpose

This document explains how vector search can check user requests and model responses against a set of guardrails.

All guardrail examples can be kept in one index. The `stage` field only tells the evaluator which guardrails apply at a particular point. It does not require separate indexes or separate algorithms.

## Guardrail Records

Each guardrail contains:

- An ID.
- A stage.
- A category.
- A description.
- One or more examples.
- An action.
- A match threshold.

Only the examples are converted into vectors. The remaining fields are stored as metadata and returned with a match.

## Preparing the Guardrails

1. Check that IDs are unique.
2. Check that the stage, action, and threshold are valid.
3. Check that every guardrail has at least one example.
4. Convert each example into a vector.
5. Store the vector with its guardrail metadata in one index.

Test cases must not be added to this index.

## Preparing Text for Search

When checking a user request, search the request itself.

When checking a model response, include the approved user request as context:

```text
User: {approved request}
Assistant: {candidate response}
```

This context is needed to detect issues such as unsuitable advice, inappropriate tone, and a response that does not address the user's goal.

## Handling Long Text

Every embedding model has a maximum input length. Text must never be silently cut off.

For long text:

1. Search the whole text when it fits within the model limit.
2. Split the text into overlapping chunks.
3. Prefer sentence or paragraph boundaries.
4. Ensure every part of the text appears in at least one chunk.
5. Search every chunk.

When checking a long response, split only the response. Add the approved request to each response chunk and ensure the combined text fits within the model limit.

If the text cannot be checked completely, return `INDETERMINATE`. Do not treat it as allowed.

## Finding Matches

Use the same process for every stage:

1. Select guardrails whose `stage` matches the current evaluation stage.
2. Compare every text view with every example in each selected guardrail.
3. Keep the closest distance found for each guardrail.
4. Match the guardrail when that distance is equal to or lower than its threshold.

Do not average distances across examples or chunks. Averaging could hide one strong match among unrelated text.

Evaluate each guardrail against its own threshold before grouping results by category. If several guardrails in the same category match, keep the strongest one for that category.

## Choosing an Action

Actions are ordered as follows:

```text
BLOCK > FLAG > ALLOW
```

If nothing matches, return `ALLOW` with the reason `no detected semantic match`.

If guardrails match:

1. Keep every matched category.
2. Choose the action with the highest priority.
3. Choose the primary match from the guardrails with that action.
4. If necessary, break ties using the match that sits furthest inside its own threshold, followed by the stable ID.

Return the final action, primary match, all other matches, distances, thresholds, and the text that triggered each match.

## Evaluation Process

```text
EVALUATE(stage, subject, context):
    select guardrails for the stage
    create complete whole-text and chunk views

    if the views cannot be created or searched:
        return INDETERMINATE

    matches = []

    for each selected guardrail:
        find its closest example across all views

        if the distance is within its threshold:
            add the guardrail to matches

    keep the strongest match in each category

    if matches is empty:
        return NO_MATCH, ALLOW

    choose the highest-priority action
    return MATCH, action, primary match, all matches
```

Before generation, the subject is the user request and no context is needed. After generation, the subject is the candidate response and the context is the approved request. Both checks use the same evaluator.

## Request Flow

1. Evaluate the user request.
2. If the result is `BLOCK` or `INDETERMINATE`, stop before generation.
3. Otherwise, generate a candidate response.
4. Evaluate the candidate response with the approved request as context.
5. If the result is `BLOCK` or `INDETERMINATE`, withhold the response.
6. Otherwise, return the response together with any flags.

## Testing

Use a separate test set that is never indexed. It should include:

- Clear matches.
- Paraphrased matches.
- Safe cases that use similar words.
- Quoted or educational material.
- Several categories in one item.
- Risks placed at different positions in long text.
- Risks that cross a chunk boundary.

Measure missed unsafe cases, incorrectly matched safe cases, category accuracy, and action accuracy. Tune each threshold separately.

## Limitations

Vector distance is not a probability or a guarantee of safety. It shows how closely text resembles the stored examples.

A missing match means only that no stored example matched within its threshold. It does not prove that the content is safe.

Vector search can identify language that resembles an unsupported claim, but it cannot determine whether the claim is true. Factual checks require trusted information.

## References

- [RedisVL SemanticRouter](https://redis.io/docs/latest/develop/ai/redisvl/0.20.0/concepts/extensions/)
- [Dai and Callan, *Deeper Text Understanding for IR with Contextual Neural Language Modeling*](https://arxiv.org/abs/1905.09217)
