# SEEK: Self-Evaluative Exploration for Knowledge Retrieval

**SEEK** is a training-free iterative retrieval and ranking framework that
addresses the single-pass limitation of conventional retrievers and rerankers.
Instead of committing to one candidate set, SEEK repeatedly cycles through
three coordinated stages:

---

## Repository layout

```
SEEK_git_repo/
├── config.yaml                  # Main configuration file
├── run.sh                       # Convenience script to run SEEK on BRIGHT
├── dataset_registry.yaml        # Benchmark registry (topics, qrels, index)
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── runner.py                # Iterative round loop (SEEKRunner)
│   ├── fusion.py                # Multi-mode score fusion (RRF, bucket, …)
│   ├── ranking.py               # Final TREC run construction
│   ├── schemas.py               # Shared dataclasses (Document, Round, …)
│   ├── query_expansion.py       # Alpha-repetition query expansion
│   ├── llm_client.py            # OpenAI LLM wrapper with disk cache
│   ├── registry.py              # Benchmark registry loader
│   └── tools/
│       ├── assessor.py          # UMBRELA-based relevance assessor (Judge)
│       ├── bm25_searcher.py     # BM25 retriever (Pyserini; default backend)
│       ├── generator.py         # Pseudo-passage generator
│       └── retriever.py         # DenseRetriever (ReasonIR-8B; optional backend)
│
├── scripts/
│   ├── common.py                # Shared CLI helpers, build_seek_runner()
│   ├── run.py                   # Main evaluation runner
│   ├── score.py                 # Score .run files with trec_eval → CSV
│   ├── write_runs.py            # Regenerate run files from existing traces
│   ├── run_artifacts.py         # Pipeline summary + evaluation metadata
│   └── embed_corpus.py          # Pre-compute corpus embeddings (offline)
│
└── prompts/
    ├── pseudo_passage_generator.yaml  # Iterative pseudo-passage generation prompt
    ├── retriever_instructions.yaml    # ReasonIR query instruction preset (structured_v1)
    └── assessor.yaml                  # UMBRELA relevance assessment prompt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your LLM API key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

The default `config.yaml` routes through [OpenRouter](https://openrouter.ai)
(`base_url: https://openrouter.ai/api/v1`). Change `llm.base_url` to `null`
for the OpenAI API directly.

### 3. Run SEEK (BM25 default)

The default `config.yaml` uses **BM25** over the Pyserini index named in
`dataset_registry.yaml`. No embedding step is required.

```bash
# Run the three default BRIGHT datasets (earth-science, economics, psychology)
./run.sh

# Run specific datasets
./run.sh bright-biology bright-robotics

# Smoke test with 5 queries
NUM_QUERIES=5 ./run.sh bright-biology

# Custom output directory
OUTPUT_DIR=/scratch/my_experiment ./run.sh
```

Per-query trace files (`.trace.json`) and TREC run files (`.run`) are written
to `outputs/` by default, or to the directory specified by `--output-dir` /
`OUTPUT_DIR`.

### Optional: switch to ReasonIR dense retrieval

To use ReasonIR-8B instead of BM25:

1. Set `searcher.backend: reasonir` in `config.yaml`
2. Uncomment the `retriever:` block and set `instruction_preset: structured_v1`
3. Set `agent.query_format: structured`
4. Pre-compute corpus embeddings once per benchmark:

```bash
python scripts/embed_corpus.py \
    --benchmark bright-biology \
    --output_dir .cache/seek_embeddings/bright-biology \
    --model_path reasonir/ReasonIR-8B \
    --batch_size 8
```

---

## Configuration

All settings live in `config.yaml`. Key sections:

| Section | Description |
|---|---|
| `llm` | LLM provider, model names, temperatures, caching |
| `searcher` | Retriever backend (`bm25` or `reasonir`), depth, BM25 overrides |
| `retriever` | ReasonIR settings (uncomment when `searcher.backend: reasonir`) |
| `generator` | Pseudo-passage generation settings (k, alpha, prompt YAML) |
| `agent` | Max rounds, termination conditions, query format |
| `fusion` | Fusion mode(s), score-0 handling, output depth |
| `eval` | Benchmark, output paths, run tag |

### Retriever backends

- **`searcher.backend: bm25`** (default) — Pyserini Lucene index from the registry.
  Use `agent.query_format: expanded`.
- **`searcher.backend: reasonir`** — ReasonIR-8B dense retrieval. Requires the
  `retriever:` block, pre-computed embeddings, and typically
  `agent.query_format: structured` with `retriever.instruction_preset: structured_v1`.



## Scoring

After a run completes, `scripts/score.py` is called automatically. To
re-score an existing run directory:

```bash
python scripts/score.py --benchmark bright-biology
```

To regenerate run files from existing traces (without reloading the model):

```bash
python scripts/write_runs.py --benchmark bright-biology
```

