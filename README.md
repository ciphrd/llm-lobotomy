# Browser Qwen ablation lab

Minimal browser-only chat and experimental MLP channel ablation using **Transformers.js 4.3.2**, Qwen3 4B, and WebGPU. No Python process, inference backend, API key, or prompt upload.

## Develop

Use Node.js 24 LTS (`nvm use`), then:

```sh
npm ci
npm run dev
```

Open the printed localhost URL. Send a message for normal chat, or enter a concept and select **Analyze**. First use downloads roughly 2.8 GB of ONNX model files, plus tokenizer/runtime assets. These are different from the previous WebLLM files. Browser caching is best-effort and requires available storage. A GPU with `shader-f16` support and substantial free GPU/system memory is required. Serve on HTTPS or localhost.

## Analysis cache

The optional Controls field accepts exactly three comma-separated words or short phrases, used verbatim after trimming instead of model-generated controls. Leave it blank for automatic selection. Duplicate controls and target/plural variants are rejected before model loading.

Completed reports are saved locally in browser Cache Storage. Analyzing the same trimmed, case-sensitive term and control choice returns the stored report before model loading or profiling. The UI shows whether results were loaded, saved, or could not be saved. Threshold selection remains interactive; applying a mask, chat, and evaluation load the model when needed. Evaluation runs are not cached.

Bump `ANALYSIS_METHOD_VERSION` in `src/inference/analysis-cache.ts` whenever profiling templates, control generation, activation collection, statistical calculations, or report schema change. On first worker use after an upgrade, all older analysis cache namespaces are deleted. Automatic controls and each explicitly supplied control list have separate entries. The key also includes model ID, pinned revision, and precision. Model download caches remain untouched. Storage denial/quota errors fall back to normal analysis; browser storage eviction can require recomputation. Cached reports from before this feature cannot be recovered automatically.

## Live 3D activations

The WebGPU view draws one point for each of the 350,208 MLP channels in 36 layer planes. Real activation frames are transferred from the inference worker after every forward pass, during both chat and analysis. Drag to orbit, scroll or use +/− to zoom, and select a layer to isolate its channels and inspect their exact values. Brightness adjusts the fixed logarithmic color scale without changing measurements.

These are absolute activation magnitudes, with masked channels displayed as zero. Single-token decoding shows the current token; prefill and teacher-forced scoring show means over the tokens in that pass. The view holds the last measurement when inference finishes and clears it when a new operation begins. It does not simulate activity or show attention neurons. Rendering requires WebGPU and uses its own GPU device; unsupported browsers and device failures show an explanatory message.

## What the experiment actually does

1. The browser downloads a pinned Qwen ONNX graph and rewrites it **in memory** before Transformers.js creates its inference session.
2. Each of the 36 MLP layers receives a dynamic 9,728-channel mask immediately before `down_proj`, after `SiLU(gate_proj(x)) * up_proj(x)`. Multiplying a channel by zero removes its contribution at every token position. Original weights stay untouched.
3. Additional graph outputs expose per-channel mean absolute activations. Transformers.js forward calls inject the masks and collect these measurements.
4. Before profiling, Qwen generates three related but distinct control concepts as a JSON array. Controls are validated for count, length, uniqueness, and simple target/plural duplication; one retry is allowed. They are not silently replaced with unrelated defaults. Semantic quality still needs human inspection.
5. The target and all three controls are run through 64 matched prompt templates (256 forward passes). Only prompt activations are collected for discovery: generated continuations do not contaminate these profiles. Each template contributes one paired observation comparing target activations with the mean of its three controls; the controls are not treated as independent replicates.
6. Every channel receives mean target/control activity, paired standardized effect size, consistency (fraction of templates with positive target–control differences), and strong-activation rate. Relatedness score = paired effect × consistency; this is not a probability. A one-sided exact sign test drops numerical ties, and BH correction covers all 350,208 channels. The UI reports how many selected channels pass q ≤ 0.05, consistency ≥ 75%, and effect ≥ 0.5, but these checks do not exclude channels from the mask. Handcrafted templates are correlated, so these statistics are exploratory.
7. All positive scores are retained. Adjust the threshold slider (logarithmic scale) or enter an exact nonnegative threshold. **Every channel with score strictly greater than the threshold** is selected, with no 512-channel cap. Zero selects all positive associations; the maximum score selects none. The initial threshold is 1. Slider changes preview the count; **Apply** updates the actual mask. The active mask is labeled separately from the current selection. Applying or restoring clears chat history.
8. Optional **Test threshold** generates original 12-token reference continuations for 12 separate prompt templates (48 baseline continuations: 12 target and 36 control) for the target and three controls, then measures the change in continuation loss for the threshold mask. It restores your previously applied mask on success, cancellation, or error. Test results include each question, original continuation, and loss change, and are labeled with their threshold and count; repeated tuning makes these exploratory tests, not independent validation. There is no automatic pass/fail or random-mask comparison.
9. Analysis restores the original model, including cancellation/error paths. Export includes all positively scored channel IDs, profiling statistics, the current threshold and selection, applied mask metadata, and the latest optional test result.

This is **an exploratory response-disruption experiment, not verified concept erasure**. Concepts are distributed; MLP channels can serve multiple functions. The small template dataset explicitly includes the target word, so lexical effects, prompt specificity, self-generated reference errors, and collateral damage remain confounds. The optional test is not a comprehensive semantic benchmark. A mask may disrupt unrelated abilities or fail to suppress the target. The tool does not ban tokens or use a system prompt to simulate forgetting. Results should be investigated with independent factual questions, paraphrases that avoid the target word, multilingual prompts, and broader capability benchmarks before drawing stronger conclusions.

## Commands

- `npm run dev`: hot-reloading Vite server.
- `npm run build`: strict TypeScript checking and production build.
- `npm run preview`: serve the production build.
- `npm test`: graph rewiring and scoring tests (synthetic fixtures; no model download).
- `npm run lint`: static lint checks.
- `npm run format` / `npm run format:check`: Prettier.

## Files

- `src/worker.ts`: serialized worker requests, stop handling, runtime lifecycle.
- `src/inference/runtime.ts`: Transformers.js loading, generation, masks, profiling, and causal trials.
- `src/inference/graph.ts`: pinned-model graph instrumentation; fails closed on unknown MLP structures.
- `src/inference/cache.ts`: caches original files and instruments the graph returned to the runtime.
- `src/inference/metrics.ts`: continuation likelihood.
- `src/inference/statistics.ts`: paired statistics, sign tests, and BH correction.
- `src/inference/probes.ts`: disjoint profiling/selection/test templates and control parsing.
- `src/App.tsx`: minimal concept input, result output, and chat.

Transformers.js is pinned because the forward adapter depends on its session behavior. The model is pinned to revision `ca652ef636e8b121ff4a1607c37ab578cb866987` of `onnx-community/Qwen3-4B-ONNX`. No modified model is uploaded. Reports contain the concept, prompts, continuations, channel IDs, scores, and evaluation results; export is a local download.

## Validation

Automated tests verify graph wiring and likelihood alignment, not semantic forgetting. Browser validation must include real model load, baseline generation, an experiment, applying/restoring a mask, and stopping a run. Masked and unmasked outputs alone are not sufficient evidence of unlearning.

## Sources

- [Transformers.js](https://huggingface.co/docs/transformers.js/)
- [Pinned Qwen ONNX model](https://huggingface.co/onnx-community/Qwen3-4B-ONNX/tree/ca652ef636e8b121ff4a1607c37ab578cb866987)
- [Qwen3 MLP implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
