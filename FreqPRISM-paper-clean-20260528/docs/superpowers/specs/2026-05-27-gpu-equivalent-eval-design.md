# GPU-Accelerated Equivalent Evaluation Design

## Context

The current UniversalFakeDetect learned-gates run is already in progress with:

```bash
scripts/evaluate_target.py \
  --target_root "dataset/UniversalFakeDetect official benchmark" \
  --output_dir results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates \
  --config_name apfreq_train100k_full.yaml \
  --device cuda:1 \
  --per_label 0 \
  --residual_batch_size 16
```

This run must remain untouched. Existing score caches and final reports from that output directory are treated as the reference experiment result.

For the GPU-equivalence gate, the baseline reference is the already-completed AIGCDetectBenchmark target report:

- Baseline result directory: `results/apfreq_full_target`
- Baseline score cache: `results/apfreq_full_target/score_cache`
- Candidate target root: `dataset/AIGCDetectBenchmark_test`

This baseline cache contains final scores, labels, and paths. It does not contain per-component scores, so the primary acceptance gate compares final scores, threshold labels, and serialized metrics against that cache. Component-level live comparisons can still be used as diagnostics, but they are not the authoritative baseline for this gate.

GPU utilization is low because the current scoring pipeline mixes GPU model forward passes with CPU-heavy image decoding, PIL transforms, tile extraction, sklearn scoring, and residual preprocessing. The enabled default path is `equivalent_fast`, which batches eligible GPU forward passes while keeping `baseline` available through runtime override.

## Goals

- Enable `equivalent_fast` by default for `apfreq_train100k_full.yaml`.
- Keep the current baseline scoring path available through `--scoring_mode baseline`.
- Keep the current in-flight experiment isolated from all optimization work.
- Add an explicit acceptance gate before an optimized path is used for benchmark reporting.
- Accept the observed `equivalent_fast` final-score drift when threshold labels and serialized metrics remain unchanged.

## Non-Goals

- Do not modify the current `universalfakedetect_learned_gates` output directory.
- Do not rewrite or invalidate existing `.npz` score caches.
- Do not change thresholds, model checkpoints, trained probes, or the final reporting contract.
- Do not silently change `residual_prior.inference_image_size` from `null` to `256`; this is only allowed if parity checks prove the final outputs are unchanged.

## Recommended Approach

Implement a two-tier evaluation mode:

1. `equivalent_fast` mode
   - Uses the same scoring semantics as the current path.
   - Batches GPU forward passes more effectively.
   - Keeps PIL-compatible transforms where GPU transforms would change pixel values.
   - Is enabled by default in `apfreq_train100k_full.yaml`.

2. `aggressive_fast` mode
   - May enable more GPU-native preprocessing or fixed-size residual batching.
   - Remains opt-in for benchmark reporting because the observed score drift is larger than `equivalent_fast`.
   - Writes its protocol metadata so reports can show that the aggressive path passed equivalence validation.

## Component Design

### Artifact Whole Scoring

The whole-image artifact path already batches tensors before GPU forward. Keep its scoring math unchanged. Only adjust batch sizing through runtime config if parity tests confirm identical or numerically equivalent scores.

### Artifact Tile Scoring

Add a result-preserving batched tile path:

- Continue using the current PIL image open, RGB conversion, native tile box calculation, tile extraction, and resize behavior.
- Accumulate tiles across images into a larger tensor batch.
- Run the artifact feature extractor on GPU in larger batches.
- Scatter tile scores back to the source image and use the existing tile aggregation function.

This avoids the current one-image-at-a-time tile GPU forward pattern while preserving the pixel inputs used by the model.

### Semantic Scoring

Use the existing batched CLIP encode structure, but keep the same `apply_variant` and `semantic_preprocess` calls. This changes batching, not image semantics.

The optimized path must produce the same per-image semantic scores as the current sequential path within tolerance.

### Residual Scoring

Keep the current residual semantics by default:

- If `residual_prior.inference_image_size` is `null`, preserve original-size inference behavior.
- Do not force resize to `256` in equivalent mode.
- Optionally group images with identical transformed tensor shapes into GPU batches. If this creates any parity failure, keep residual scoring sequential for equivalent mode.

Aggressive residual resizing or GPU-native preprocessing is allowed only behind the parity gate.

## Equivalence Gate

Add a parity check command or test that compares the completed `results/apfreq_full_target/score_cache` baseline against optimized scoring on a fixed sample set:

- Multiple AIGCDetectBenchmark generators.
- Both `0_real` and `1_fake` samples.
- Small, medium, and high-resolution images.
- At least one group with native tiling behavior.

The parity check must compare:

- Final scores from the baseline cache versus candidate `final_fixed` scores.
- Predicted labels at the configured threshold.
- Per-generator metrics for every sampled generator.
- Aggregate report rows for a small temporary output directory.

Acceptance criteria:

- Predicted labels must match exactly.
- `final_fixed` max absolute difference must be `<= 1e-3` for the enabled `equivalent_fast` path.
- Per-generator and aggregate metrics must be identical after normal CSV numeric serialization.
- Aggressive mode may be approved only if the user explicitly accepts its larger measured score drift.

## Runtime and Protocol

Add explicit runtime metadata to `protocol.json` for optimized runs:

- scoring implementation mode: `baseline`, `equivalent_fast`, or `aggressive_fast`.
- whether the parity suite was run.
- parity suite command and summary.
- optimized component flags used for artifact, semantic, and residual scoring.

Future benchmark output directories should be separate from the current run unless the user explicitly requests reuse.

## Testing

Add focused tests around:

- Tile batching scatter/aggregation correctness.
- Semantic batched scoring parity.
- Residual same-shape batching, if implemented.
- End-to-end parity on a small file-backed sample set.

The implementation is complete only after:

- The `results/apfreq_full_target` baseline result files exist.
- The optimized path passes the parity gate.
- A trial optimized run writes protocol metadata that proves which mode was used.
