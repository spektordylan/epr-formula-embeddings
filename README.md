# epr-formula-embeddings

Second in a series of ML4TP mini-projects. Goal: embed logical formulas (from
TPTP's EPR/effectively-propositional division) and see whether similar
formulas end up close together.

## Pipeline

1. `scripts/batch_parse.py` - diagnostic pass over raw .p files, prints stats
2. `scripts/build_corpus.py` - parses + filters to confirmed EPR problems, writes `data/processed/corpus.json`
3. `scripts/build_features.py` - structural feature baseline (literal counts, symbol bag), writes `data/processed/clause_vectors.npy`
4. `scripts/build_gnn_embeddings.py` - graph-based embeddings via an untrained message-passing GNN, writes `data/processed/gnn_vectors.npy`
5. `scripts/eval_retrieval.py <vectors_file>` - nearest-neighbor retrieval quality vs. chance baseline
6. `scripts/prepare_viz_data*.py` + `scripts/build_viz_html.py` - interactive click-through similarity explorer (self-contained HTML)

Raw data isn't committed - see below for how to pull the EPR subset from TPTP and regenerate everything above.

## Results

The structural baseline (literal and variable counts, symbol frequency) mostly learns which generator wrote a clause, not what it means. The same-domain retrieval rate was 94%, against a chance baseline of 22%, which looks good until the actual results are examined: a Geometry clause's nearest neighbors were Hardware Verification clauses with meaningless auto-generated symbol names. Different TPTP domains have different authoring styles: synthetic domains use placeholder predicates like `p` and `q`, while hardware and software verification problems are bit-blasted and use names like `ssSkP12`. That stylistic fingerprint is what the baseline was actually matching on. This is easy to see in the interactive explorer: clicking through neighbors tends to get stuck in a single domain, a kind of spider trap, without ever leaving that cluster.

Filtering down to the roughly 8,000 clauses with genuinely human-chosen names (Puzzles, Geometry, NLP, Group Theory, and similar) removes the worst of this generator-style confound and gives a smaller, more legible picture.

As a fix, a symbol-name-invariant graph representation was built. Each clause becomes a small graph, with literal nodes, variable nodes, and constant nodes, connected by edges that record argument position. Node features describe structure only, never which predicate or constant a node actually represents. This works as intended: `~p(X,Y) | q(X)` and `~same_day(A,B) | is_weekday(A)` produce identical embeddings.

This was run untrained, using fixed random weights, as a first pass, since no labeled similarity data exists yet to train against. Along the way, a real bug was found: summing incoming messages instead of averaging them made embedding magnitude track clause size and degree rather than actual structure, which collapsed the 2D PCA projection onto a single dominant dimension. This was fixed by mean-aggregating and L2-normalizing the output.

Even after the fix, the untrained GNN's same-domain rate (87%) is hard to interpret. It is lower than the structural baseline, but that could mean either less confound-driven matching or simply noisier embeddings, and there is no way to distinguish these without training. That is where this stands for now.

## Status: paused, not finished

Training the GNN was not pursued here. A contrastive objective, where clauses from the same problem should embed closer together than random pairs, is the obvious next step, but it is a substantial amount of additional work. Premise selection, the next mini-project, has a much stronger supervision signal available: actual proof usage data, rather than a weak self-supervised proxy. The graph representation and encoder architecture carry over directly, so the training effort is better spent there.

## Setup

Raw data is not included in this repo. To source it:

1. Download the TPTP library from https://tptp.org/TPTP/Distribution/TPTP-v9.2.1.tgz (881MB compressed).
2. Extract the `Problems/` directory and filter to EPR-tagged files using the SPC header field:
```bash
   mkdir -p epr_subset
   grep -rlE '% SPC\s*:\s*(FOF|CNF)_(SAT|UNS)_EPR' TPTP-v9.2.1/Problems | xargs -I{} cp {} epr_subset/
```
3. Place the resulting files under `data/raw/`.
4. Run the pipeline in order (see above) starting with `scripts/build_corpus.py`.

Everything downstream needs `numpy` and `scikit-learn` (for PCA in the viz prep step).