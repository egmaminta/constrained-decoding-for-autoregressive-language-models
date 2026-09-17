# Constrained Decoding for Autoregressive Models

Constrained decoding (or constrained candidate scoring) uses an autoregressive language model to choose from a fixed set of tokens. This notebook loads `Qwen/Qwen2.5-1.5B-Instruct` and reuses the KV cache after processing the shared prompt.

The model scores the allowed values for each field. Python puts the selected values into a dictionary. This is closer to classification than full JSON generation.

## Multi-token collisions

Multi-token choices can collide when they begin with the same token. I tried a trie fallback that followed the shared tokens until the choices split.

The test prompt said the sky would stay clear and sunny all day. The model had to choose from these values.

- `not sunny`
- `not rainy`
- `sunny`

The right answer was `sunny`, but the trie ranked `not sunny` first.

Both negative choices started with `not`, so the fallback kept scoring that branch to separate `sunny` from `rainy`. The standalone `sunny` choice kept only its first-token score. The final ranking mixed scores built from different numbers of tokens. That made the result unreliable.

The notebook now scores complete candidate values. It also keeps the letter-label experiment from the original work.

## Shuffled letter labels

The multi-token choices were replaced with short labels. I avoided the usual `A`, `B`, and `C` labels because models can prefer familiar answer positions. I used `Z`, `X`, `W`, and `Q`, then changed the mapping between runs.

| run | mapping | winner |
|---|---|---|
| 1 | `Z = not sunny` · `X = not rainy` · `W = sunny` | `W = sunny` |
| 2 | `Q = not rainy` · `W = not sunny` · `Z = sunny` | `Z = sunny` |
| 3 | `X = sunny` · `Q = not sunny` · `W = not rainy` | `X = sunny` |

The winning letter changed, but the decoded answer stayed `sunny`. This suggests the model followed the mapping instead of choosing one favorite letter. Repeat the check after changing the model or prompt.

## Structured generation in an application

For most application code, a better starting point is a Pydantic model that describes the expected output. Pydantic handles types and validation, and it can produce the JSON Schema used by a structured-generation backend.

Pydantic does not constrain model tokens by itself. Pass its schema to a runtime that supports JSON grammar or native structured output. Parse the result with the same Pydantic model before using it.

This notebook is a lower-level experiment for fixed candidate lists. It is useful for studying logits and KV-cache reuse, but it is not a replacement for schema-based structured generation.

## Running the notebook

Install recent versions of `torch`, `transformers`, and `accelerate`. Open [constrained-decoding.ipynb](constrained-decoding.ipynb) and run the cells in order.

The notebook has no saved outputs. Model and package updates can change the scores.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).

Maminta, Emmanuel. *Constrained Decoding for Autoregressive Models*. 2026. GitHub repository.

```bibtex
@software{maminta2026constrained,
  author = {Emmanuel Maminta},
  title = {Constrained Decoding for Autoregressive Models},
  year = {2026},
  license = {MIT}
}
```

## License

Released under the [MIT License](LICENSE).
