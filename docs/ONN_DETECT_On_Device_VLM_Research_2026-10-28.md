**ONN-DETECT: On-device VLM selection, Colab T4 training, and detection research**

Prepared for Shubham Kumar · 28 October 2026, India Standard Time

Repository inspected: [shubhamiitshk/Onn-Detect-](https://github.com/shubhamiitshk/Onn-Detect-), public main at [3da11a3d13690a1d3c550410e797030517a8756b](https://github.com/shubhamiitshk/Onn-Detect-/commit/3da11a3d13690a1d3c550410e797030517a8756b), committed 26 October 2026. The recursive tree contained 454 entries.

**My recommendation is to train Qwen3.5-2B as the first quality-focused forensic VLM candidate on a Colab T4, and evaluate it as an optional second-stage model alongside your existing detectors.** Keep Qwen3.5-0.8B as the smaller deployment candidate and SmolVLM2-500M-Video-Instruct as the compact baseline. Choose the shipped model by additional recall at a fixed false-positive rate, memory, and latency on real phones. None has established “world-class” forensic performance in your app merely by being a capable general VLM.

This is a research and source review. I did not train these models, benchmark an Android phone, inspect your local Windows checkout, or measure a release APK. Published model artifacts, vendor notebook outputs, repository-recorded results, and my proposed engineering choices are identified separately below. No repository files were changed.

**The model decision depends on the download you can accept.** I assume the existing Android app remains offline during analysis and that a VLM is an addition to the current pipeline. Device tiers below are initial product targets, not tested minimum requirements.

| Product constraint | First candidate | Intended role |
|---|---|---|
| Quality-focused deep scan; roughly 2 GB optional model pack is acceptable; initially target 8 GB+ phones | Qwen/Qwen3.5-2B | First model to fine-tune and evaluate for a useful second opinion |
| Smaller pack; initially target 6 GB+ phones, then test lower tiers | Qwen/Qwen3.5-0.8B | Middle option with a published pack of roughly 738 MB in the configuration below |
| Very small-model experiment and broad device reach | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | Compact baseline and possible student; ship only if it adds value |
| Google-native multimodal runtime is the priority and training hardware can expand | google/gemma-4-E2B-it | Alternative worth testing; the official vision-tuning recipe is not a T4 recipe |
| Existing pipeline already meets the product’s quality and resource goals | Keep specialist models alone | A VLM must earn its additional download, RAM, and wait time |

The primary reason to start with Qwen3.5-2B is practical: its maintained training ecosystem includes a vision notebook explicitly intended for a free T4, with saved T4 execution output. This establishes a credible starting workflow. It does not establish a forensic accuracy advantage over the other candidates. [S1, S2]

**Several claims in the pasted answer need correction before implementation.** The public repository and primary sources support a more cautious interpretation.

| Quoted claim | What I could establish | Consequence |
|---|---|---|
| VLM training script and Android analyzer were created and tested | Neither train_forensic_vlm_colab_t4.py nor OnDemandForensicVlmAnalyzer.kt exists anywhere in the inspected public tree | They may be unpushed local work. The pasted command is not runnable from this public revision |
| DeviceCapabilityManager.kt and SiglipSemanticAnalyzer.kt were inspected | Those filenames are also absent; the current relevant analyzer is VfmProbeAnalyzer.kt | Use the actual public interfaces when planning the addition |
| DeepThinkAlgorithmsTest verifies the VLM selector and rubric head | The public test file covers texture contrast, hotspot localization, rPPG helper logic, and video score policies | Those tests do not establish VLM integration or quality |
| Public release is v11.3 | The current README and release manifest describe onn-prod-v10 / v10.1 | Do not mix local version claims with public release evidence |
| 190 ms verdict, 520 ms explanation, 590 MB RAM, and 28.5% escalation are exact | I found no matching end-to-end VLM measurements in the inspected public source and artifacts | Treat them as unsupported estimates until reproduced |
| SigLIP and the VLM can never coexist in active memory | LiteRTInferenceEngine caches loaded interpreters; it has an engine-wide close operation and no per-model eviction method | Sequential execution does not establish sequential residency |
| An mmap model has effectively zero memory cost | Mapping avoids a full Java byte-array copy, but resident pages, native buffers, packed weights, activations, and accelerator allocations still matter | Profile whole-process memory rather than Java heap alone |
| A 46 KB rubric projection is an established forensic breakthrough | A small projection is plausible, but I found no validated implementation or detector benchmark for it in public main | Treat it as a hypothesis requiring training, export support, and ablation |
| SmolVLM2-500M-Instruct is the model identifier to use | The official 500M video model is HuggingFaceTB/SmolVLM2-500M-Video-Instruct | Use the exact identifier and its matching processor/template |
| GRPO gains and exact T4 memory from larger forensic papers transfer to 500M | The papers use different backbones, datasets, rewards, and training hardware | Reproduce a small experiment before committing to RL |

Source evidence for the repository-specific findings is linked as R1–R9. The absence claim concerns this pinned public tree, not files on your Windows machine.

**Your own saved experiment argues against deploying an untrained VLM as the final detector.** The repository’s llm_signal_analysis_results.json records the following older experiment:

| Experiment entry | Recorded ROC-AUC |
|---|---:|
| Zero-shot SmolVLM-256M | 0.58925 |
| Zero-shot SmolVLM-500M | 0.549625 |
| Trained probe using the SmolVLM 500M vision tower | 0.8692 |
| Trained MobileCLIP2-S0 probe | 0.885075 |

The experiment script identifies the zero-shot models as the original SmolVLM models, not SmolVLM2, and applies a common image-laundering procedure. These are repository-recorded results that I did not rerun. They do not settle the performance of a fine-tuned SmolVLM2 or Qwen model, and they should not be compared directly against v10 metrics from a different evaluation setup. They do show why “the model can describe artifacts” is insufficient evidence that it detects synthesis. [R5, R6]

There is also an immediate validation issue: release_manifest.json records desktop_validated=true, validated=false, and device_validation.passed=false. The release still requires independent device evaluation according to its own manifest. Adding a new VLM should not bypass that gate. [R2]

**The current research supports forensic supervision, complementary evidence, and modern testing.** It does not identify one universally best mobile model.

| Research | Useful evidence | Implication for ONN-DETECT |
|---|---|---|
| FakeVLM / FakeClue | A specialized detection-and-explanation workflow; the authors’ code starts from LLaVA-1.5-7B and links the released FakeClue data | Reuse suitable training annotations and evaluation ideas; the full research model is not the first mobile candidate [S3, S4] |
| ThinkFake | Uses Qwen2.5-VL-7B and a structured training/reward process, including specialist detector supervision | A 500M model with a prompt is not a reproduction of ThinkFake [S5] |
| FakeVLM-R1 | Uses SFT plus GRPO and authenticity counter-evidence; its reported training/evaluation uses eight H200 141 GB GPUs | Learn from real-image counterexamples, but do not transfer its hardware figures or gains to T4 [S6] |
| LEGION | Studies detection, localization, and explanation together | An explanation should be evaluated against image regions, not just fluent prose [S7] |
| B-Free | Uses real-conditioned synthetic counterparts to reduce source and semantic biases | Pairing and balanced processing deserve as much attention as model architecture [S8] |
| DailyBench v6, updated 22 October 2026 | Reports methods with 91–96% GenImage balanced accuracy falling to 52–79% on modern full synthesis and 43–67% on localized manipulations | Old benchmark success does not establish current generalization [S9] |
| TextFake | Evaluates text-rich synthetic images and reports that none of its evaluated methods exceeds 80% accuracy | Documents and screenshots need an explicit benchmark and route [S10] |
| WildFakeToolBench, 22 October 2026 | Examines image-processing robustness; the study is small and detector-specific | Include processing chains in robustness evaluation, without treating this paper as a complete ranking [S11] |

The most transferable idea from FakeVLM-R1 is to train on authentic explanations and counterexamples as well as fake artifacts. Natural bokeh, difficult reflections, low-light denoising, and sharpening can resemble artifact descriptions. The paper’s improvements are evidence for its own setup, not a promised 11–18% improvement for ONN-DETECT.

**I would use the following model shortlist.** The role column is my recommendation; it is not a claim of measured detector superiority.

| Model | Strength relevant to this task | Main trade-off | My assessment |
|---|---|---|---|
| Qwen3.5-2B | Image/text model with current vision tuning and GGUF workflows | Considerably larger vision-language pack than SmolVLM2 | First quality candidate |
| Qwen3.5-0.8B | Smaller current vision-language model with a T4 training path | Lower capacity; the vision component must still be included | First smaller-pack challenger |
| SmolVLM2-500M-Video-Instruct | Compact image/video model; the authors provide a Colab fine-tuning example | Fine-grained forensic discrimination is unproven in your app | Essential compact baseline |
| Qwen3-VL-2B-Instruct | Useful established Android deployment alternative, including Arm guidance | Adds another training/deployment family | Fallback if the Qwen3.5 path causes material export or runtime problems |
| Gemma 4 E2B | Current on-device-oriented image/audio model with LiteRT-LM support | The official vision fine-tuning guide asks for BF16-capable hardware and more than 16 GB | Strong alternative when the training constraint changes |
| LFM2.5-VL-1.6B | Edge-oriented model with official GGUF/ONNX formats and configurable image processing | Different runtime/training trade-offs and model license | Speed-focused challenger after the first pilot |
| SmolVLM2-256M | Extremely small | Less capacity to learn subtle forensic cues | Use for a strict size experiment, not the primary quality target |

Qwen’s published general vision benchmarks help motivate a capacity comparison, but they are not forensic benchmarks. Liquid’s card similarly reports general vision/OCR results. I would not select a detector by MMMU, OCRBench, parameter count, or generation fluency alone. [S12–S15]

For the Google route, compare Gemma 4 with the earlier Gemma 3n suggestion. Gemma 4 E2B supports image and audio input, but “effective 2B” is not a complete physical weight count. Google lists 2.3B effective and 5.1B with embeddings. Its optimized mobile formats have different loading behavior from a generic Q4 export. The official vision fine-tuning recipe explicitly requires BF16-capable hardware such as L4/A100 and more than 16 GB, so that recipe does not satisfy your T4 constraint unchanged. This is a tooling reason to start elsewhere, not proof that Gemma cannot be adapted to a T4. [S16, S17]

**The size calculation must include the vision model.** llama.cpp normally uses a language GGUF plus its matching mmproj component, which includes image encoding and projection. A language-only file cannot inspect an image. Published rounded file sizes below are reference artifacts, not the final size of your future fine-tune. MB and GB in this table are decimal. [S18]

| Reference configuration | Language file | Vision/projector file | Approximate combined pack |
|---|---:|---:|---:|
| SmolVLM2-500M, maintainer Q8_0 files | 437 MB | 109 MB Q8_0 | 546 MB |
| Qwen3.5-0.8B, Unsloth Q4_K_M + FP16 mmproj | 533 MB | 205 MB | 738 MB |
| Qwen3.5-2B, Unsloth Q4_K_M + FP16 mmproj | 1.28 GB | 668 MB | 1.95 GB |

These rows intentionally specify their different quantization configurations. They are a storage comparison, not an accuracy comparison at identical precision. SmolVLM2’s FP16 projector is listed at 199 MB; choosing it instead of Q8 changes the total. A smaller custom quantization is an experiment to validate, not a size to promise in advance. [S19–S21]

Do not infer peak phone RAM from these downloads. Runtime memory also includes image decoding, visual-token count, context/KV or recurrent state, compute buffers, native runtime state, packed weights, and GPU/NPU allocations. An optional external pack keeps its weights outside the APK, but the new native runtime and integration code can still increase the APK. Installation, staging, updates, and rollback can require additional disk space.

There is a smaller unit issue in the current claims: the public SigLIP LFS pointer declares 94,329,440 bytes. That is 94.33 MB or approximately 89.96 MiB. It is not evidence for the quoted universal APK size or installed app size; those require the actual build artifact and install. [R10]

**My proposed algorithm keeps three different kinds of evidence explicit.**

1. Verify available provenance and input integrity. Validate a signed claim and its media binding before treating it as evidence of AI involvement. Missing credentials are unknown. Differentiate a claim of AI editing from a claim of complete generation.
2. Run the current specialist detectors. Preserve the existing SigLIP/TAP and SRM/PatchCraft branches as the image baseline. Route audio to the anti-spoofing pipeline and video to both visual and temporal/audio analysis.
3. Select whether the VLM is worth running. Candidate triggers include detector disagreement, uncertainty, a suspected local edit, or a document/screenshot route. Learn thresholds on held-out data and measure the resulting escalation rate.
4. Give the VLM the original global image and, when useful, one bounded crop from the original image. Keep a crop coordinate transform. A residual heatmap may propose a region, but should not be substituted for the natural image as the model’s only visual input.
5. Produce a trained classification score and a small set of grounded observations. Each observation should have a region or an explicit “not localized” status. Keep content type separate from authenticity.
6. Combine the VLM measurement with existing measurements using a newly trained, calibrated decision layer. Include unavailable/model-skipped state. Evaluate correlated errors because your current semantic detector and the VLM may share similar features.
7. Return a calibrated assessment, an inconclusive option, and a short evidence summary. Report provenance separately. A fluent explanation does not make a detection probability a proof.

This is my engineering proposal. Its additional detection benefit is unknown until tested. Start with the VLM in a diagnostic or shadow role so errors do not silently change the public detector’s operating point.

**A six-axis head is a reasonable experiment only when its labels and runtime contract are real.** Candidate axes could describe texture, boundaries, geometry, illumination/reflections, text layout, and biological appearance. They need not all apply to every image. Train with “not applicable” and “not observable” states, and mask missing annotations in the loss.

Do not create six “authentic counter-logits” by mechanically reversing fake labels. An authentic bokeh example requires a different annotation from a fake-boundary example. Likewise, a model’s verbal confidence is not a calibrated probability.

For a simpler first pilot, train a short, fixed class response and score its actual token likelihoods, checking tokenization. Alternatively train a supervised binary/multiclass head if the intended mobile runtime exposes the required representation. Add auxiliary artifact labels only when the dataset supports them.

A last-prefill-state head can avoid long autoregressive decoding, but it still pays the vision and prompt-processing cost. It does not perform an explicit multi-step counterfactual analysis merely because the projection is called “dialectical.” The official SmolVLM2 config lists decoder width 960; twelve FP32 outputs plus bias would occupy 46,128 bytes. That arithmetic explains a small head size, not its accuracy or latency. Qwen3.5-2B uses a different hidden width, and a custom head also needs export and native invocation support. [S22, S23]

If you later generate an explanation, use a short evidence summary tied to the actual measured signals. Evaluate whether it changes when a cited region is removed and whether it hallucinates artifacts on authentic photographs. Avoid treating a generated internal narrative as a forensic audit trail.

**For Colab T4, start with supervised LoRA and establish the export path first.** Use the maintained [Unsloth Qwen3.5-2B vision notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_%282B%29_Vision.ipynb) as the starting environment, then replace its demonstration data with the forensic task. Keep the notebook/model/library revisions in your experiment record.

The inspected vendor notebook explicitly targets a free Tesla T4. Its saved example reports BF16=false, a 14.563 GB device, and 5.146 GB peak reserved CUDA memory for its demonstration run. That is useful feasibility evidence. It is not an end-to-end memory measurement for your dataset, resolution, dual-image inputs, or GRPO workload, and I did not reproduce it. [S2]

The current Unsloth Qwen3.5 guide discourages 4-bit QLoRA because of quantization differences and provides 16-bit LoRA workflows. On T4 use the supported FP16 path rather than forcing BF16. Training precision and Android inference precision are separate choices: train/adapt in 16-bit, merge appropriately, then compare a quantized export against that reference. Do not assume the standard Ampere-oriented FlashAttention-2 installation works on T4; use the notebook’s tested path. [S1, S24, S25]

| Setting | Proposed starting choice | Reason |
|---|---|---|
| Base model | Qwen3.5-2B | First quality experiment |
| Adaptation | 16-bit LoRA; FP16 on T4 | Avoid relying on unsupported native BF16 |
| LoRA rank | 8 or 16 | Small pilot before increasing capacity |
| Microbatch | 1 | Leave room for image-dependent activation peaks |
| Gradient accumulation | 8–16 | Increase effective batch without simultaneous image batches |
| Sequence cap | Start at 1,024 total tokens; verify actual visual-token usage | A text cap alone does not bound multimodal memory |
| Image input | One global image first | Establish a working baseline before adding a crop |
| Image budget | Begin around 384–512 px, with an explicit visual-token/tiling cap | Proposed speed/detail trade-off to tune |
| Vision adaptation | Compare frozen vision against LoRA on selected final vision blocks | Determine whether language-only adaptation misses useful forensic cues |
| Gradient checkpointing | Enable the supported notebook setting | Reduce activation memory |
| Training cache | Disable autoregressive cache when required by the training implementation | Avoid unnecessary cache allocations |
| Output | Class plus short, observable evidence | Limit training and generation length |
| Pilot | About 1,000–2,000 unique examples, then a larger disjoint study | Detect data/runtime failures before a long run |
| Initial full study | Roughly 10,000–30,000 carefully curated unique examples | A planning scale, not a promised optimum |
| Epochs | Start with one; choose by held-out performance | Avoid declaring two epochs universally optimal |
| RL | Off in the first study | Establish whether supervised adaptation already adds useful recall |

These values are my proposed starting configuration. I have not benchmarked their VRAM or throughput. Run a 20–50 optimizer-step smoke test that includes the largest permitted images, then record actual peak allocated and reserved CUDA memory, tokens, and examples per second. Leave several GB of measured headroom before trying the second image. Resume from checkpoints because free Colab availability and session duration are not guaranteed.

**The data work is likely to matter more than choosing between two neighboring model sizes.** Use four complementary ingredients:

- Authentic photographs from diverse phone cameras and processing modes, with difficult negatives such as bokeh, HDR, low light, denoising, blur, sharpened skin, reflections, and screenshots of real photos.
- Fully synthetic images from diverse, recent generator families. Include current generators beyond the older GenImage set, and hold out entire generator families.
- Localized edits with real source images and masks when available: inpainting, object replacement, face edits, and generated text. Preserve the original-to-edited relationship in the split metadata.
- Licensed detection-and-explanation examples such as FakeClue, plus reviewed annotations that identify both visible evidence and legitimate real-image alternatives.

FakeClue is released by the authors and includes material drawn from GenImage, FF++, and Chameleon. Therefore, training on FakeClue and testing on those datasets is not automatically independent. Deduplicate underlying source images and their derivatives, rather than relying only on dataset folder names. I confirmed the original FakeClue release; I did not verify a downloadable FakeClue++ release during this review. Do not make the first training run depend on an unverified download. [S3, S4]

B-Free’s pairing principle is useful: condition synthetic examples on real counterparts and balance semantic content and processing. Keep each real source, its reconstructions, crops, edits, and all degraded versions in one split. JPEG/PNG choice, resolution, watermarking, padding, and file naming must not become class shortcuts. Paired reconstructions are one ingredient, not the entire training distribution. [S8]

For explanations, annotate what is visible. A detector score can supervise classification, but it cannot by itself justify a sentence about a malformed finger or impossible reflection. Review teacher-generated annotations; keep uncertainty and a “no visible artifact” option. Some generated images lack a reliable visible semantic defect even when a specialist finds statistical evidence.

**Use a controlled evaluation before calling this a detection upgrade.** Freeze the data and operating points before the final evaluation.

| Candidate | What the experiment isolates |
|---|---|
| A: Exact current quantized pipeline | Baseline |
| B: Current pipeline plus untrained VLM measurement | Whether general prompting adds anything |
| C: Current pipeline plus forensic SFT/LoRA VLM | Incremental benefit of forensic training |
| D: C plus one selected original-image ROI | Benefit and cost of the crop |
| E: C plus a supervised auxiliary head | Whether the proposed head adds quality or only speed |
| F: Smaller student or smaller backbone | Mobile size/quality trade-off |
| G: GRPO candidate, only after C succeeds | Whether RL adds reproducible value beyond supervised adaptation |

Evaluate A–G on the same held-out sources and transformations, with a budget-matched version where appropriate. Include VLM-only scores to understand whether an ensemble is adding complementary evidence. Train the fusion layer using out-of-fold or otherwise properly held-out measurements, and calibrate on data not used to fit it. A router changes the distribution seen by the second stage; assess calibration and performance on both the routed subset and the complete pipeline.

Use original held-out tests from OpenFake, suitable deduplicated LOKI and other explanation benchmarks, DailyBench’s full-synthesis and manipulation subsets, and TextFake. Verify each release, license, and overlap with your training corpus before use. Maintain separate phone-camera and social-sharing tests relevant to your actual users. The first pilot can use a small diagnostic subset, but final generalization claims need disjoint data. [S9, S10, S26]

Measure ROC-AUC and average precision, but prioritize TPR at 1% and 0.1% FPR where enough authentic samples exist. Also report per-source/per-generator results, real-image false-positive rates, edit-size/localization results, uncertainty coverage, and paired confidence intervals for the difference from baseline. Do not silently exclude inconclusive cases.

A small balanced test cannot support precise ultra-low-FPR claims. As an illustration, 200 real images give an empirical false-positive count increment of 0.5 percentage points. Plan thousands to tens of thousands of distinct real examples for meaningful low-FPR assessment, report binomial confidence intervals, and account for correlated derivatives. This is a sample-size planning recommendation, not a certification threshold.

For robustness, apply processing to both classes: JPEG recompression, repeated sharing/re-encoding, resizing, crops, screenshots, blur, denoising, overlays, and format changes. Keep all derivatives of a source together when resampling confidence intervals. A model that recognizes which class was encoded in which format can look excellent and fail immediately in a phone app.

**I would postpone GRPO until it passes a clear decision test.** It adds rollout generation, optimizer state, retained activations, and potentially reference-policy and reward-model costs. Group size alone does not determine memory. Short sequential rollouts may be feasible, but they also change throughput and the training implementation.

If SFT produces measurable benefit, try a bounded experiment with a small group, short outputs, and fixed image budgets. Reward correct classification and genuinely grounded observations; use authentic false positives as explicit counterexamples. Compare against an SFT run with the same data and similar compute. A formatting reward or verbose counterargument can improve the reward without improving detection.

Distillation may be a more useful second investment for mobile: train the quality candidate, then transfer its useful scores or reviewed annotations into a smaller VLM or a specialist vision student. The shipped student still requires its own held-out and quantized evaluation. Do not assume a teacher’s accuracy transfers automatically.

**The Android integration needs a separate VLM runtime contract.** The current LiteRTInferenceEngine accepts one float input tensor, requires one scalar output, validates that output as a probability, and caches loaded interpreters. A tokenized VLM with image encoding, decoder state, and optional generation is not a drop-in TFLite classifier. [R3]

My first deployment experiment would use a pinned llama.cpp build with its multimodal support, accessed through an Android native/JNI wrapper. It has documented Android builds and separates the language model from the image encoder/projector. An alternative is LiteRT-LM for a supported model, especially the Google route. Confirm the specific fine-tuned model, processor, image path, and backend together; text-only loading is not a successful vision export. [S18, S27, S28]

Before a large fine-tune, export a tiny adapter experiment and prove all of the following:

- The mobile runtime loads both language and matching vision components.
- Identical image/prompt fixtures produce comparable scores to the reference implementation.
- Changes in processor, image tiling, normalization, chat template, or class-token mapping are versioned.
- Quantization retains useful recall at the selected false-positive rate.
- Missing, wrong, or incompatible pack components fail with an unavailable result.
- Cancellation, repeated scans, and native resource release work on actual devices.

If vision/projector weights are adapted, export those adapted components too. Reusing the original mmproj with a fine-tuned vision tower can silently invalidate the result. If a custom classifier head is used, it is another signed artifact with a defined tensor/representation contract.

Your current Android manifest removes INTERNET and ACCESS_NETWORK_STATE. The simplest compatible pack workflow is user download outside the app followed by signed local import through a document picker. Validate a trusted publisher signature, every file hash, declared size, runtime/frontend version, and safe archive paths before activation; stage then atomically activate, with a bounded rollback policy. This is a proposed extension, not a claim that public main already implements a signed VLM pack. [R7]

A real memory policy must manage ownership of the current cached interpreter and the VLM session. Close or evict large specialists when needed, clear decoder state when finished, bound concurrent scans, and release on memory pressure. Measure whether repeated cold loading costs more time and energy than a short bounded foreground session. Neither sequential function calls nor System.gc() establishes native-memory release.

Use Android’s memory tools to record PSS/RSS, native and managed memory, and relevant accelerator allocations. Test cold load, warm scan, global-plus-ROI input, optional explanation, repeated scans, and cancellation. Report SoC, OS, backend, runtime commit, threads, token/image budgets, thermal conditions, p50/p95 latency, and peak memory. Phone RAM capacity alone is an inadequate compatibility rule. [S29]

**Other media should use dedicated forensic evidence even when a VLM understands the content.**

| Input | Existing or proposed primary route | Useful VLM contribution | Limitation to preserve |
|---|---|---|---|
| Photographs | SigLIP/TAP + residual detector + calibrated fusion | Semantic inconsistencies and grounded regional observations | Plausible appearance can coexist with AI generation |
| Local image edits | Original-image ROI and manipulation/localization training | Describe the edited region and surrounding inconsistency | A whole-image generation score can miss a tiny edit |
| Faces | Face-specific forensic training, crop quality checks | Describe observable facial artifacts | Unusual appearance alone is not an authenticity test |
| Videos | Visual frames + temporal/face/audio-video evidence | Sparse-frame or short-clip semantic checks | Captioning video does not validate temporal forensics |
| Speech/audio | LCNN baseline and separately evaluated anti-spoofing upgrades | Explain measured findings; optional transcription for a separate purpose | ASR/audio understanding does not establish clone detection |
| Documents/receipts | Document-specific tampering tests, layout/OCR consistency, provenance | Read and compare visible text/layout details | It cannot establish that a bank transaction occurred |
| Provenance | Signature and media-binding verification | Explain verified claims | Missing credentials do not establish human origin |

For audio, evaluate future anti-spoofing candidates against your current LCNN on codec-degraded, speaker-disjoint, and in-the-wild data. For video, keep clips and identities disjoint across splits. Neither route should inherit the image detector’s calibration or accuracy claims.

**I would carry out the next work in this order.**

1. Reconcile the quoted local VLM work with public main if those local files exist. Keep a pinned baseline and complete the device validation already required by the manifest.
2. Prove Qwen3.5-2B vision inference and export on a representative Android phone before a long training run.
3. Build a deduplicated pilot with authentic hard negatives, modern synthesis, and local edits. Establish a compact external test.
4. Run the T4 LoRA pilot, initially with one image. Compare the resulting model against current specialists and the untrained VLM.
5. Train/evaluate Qwen3.5-0.8B and SmolVLM2-500M under the same data and evaluation rules. Use the smallest model meeting the product’s measured goal.
6. Add ROI input and a supervised head only when the corresponding ablation justifies them.
7. Refit and recalibrate the full routed pipeline, including missing-pack fallback, then verify the quantized mobile artifacts.
8. Consider distillation or GRPO after the supervised baseline and deployment budget are established.

The concrete first choice is Qwen3.5-2B for an optional quality pack, with Qwen3.5-0.8B as the likely smaller-pack challenger and SmolVLM2-500M as the compact baseline. The product improvement to seek is fewer missed modern fakes and local edits at the same false-positive rate within a measured phone budget.

**Primary sources and inspected repository evidence follow.** URLs were accessed for this review on 28 October 2026 IST. Research results belong to their published settings; repository results were read rather than rerun.

| ID | Source |
|---|---|
| S1 | [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) |
| S2 | [Official Qwen3.5-2B vision notebook, including saved T4 execution outputs](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_%282B%29_Vision.ipynb) |
| S3 | [FakeVLM author repository and training/data links](https://github.com/opendatalab/FakeVLM) |
| S4 | [Authors’ FakeClue dataset](https://huggingface.co/datasets/lingcco/FakeClue) |
| S5 | [ThinkFake paper](https://arxiv.org/html/2509.19841v1) |
| S6 | [FakeVLM-R1 paper](https://arxiv.org/html/2605.30062v1) |
| S7 | [LEGION, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.pdf) |
| S8 | [B-Free, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf) |
| S9 | [DailyBench, v6 updated 22 October 2026](https://arxiv.org/html/2607.24016v6) |
| S10 | [TextFake](https://arxiv.org/html/2606.01050v1) |
| S11 | [WildFakeToolBench](https://arxiv.org/abs/2610.31152) |
| S12 | [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) |
| S13 | [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| S14 | [SmolVLM2 author article and Colab training link](https://huggingface.co/blog/smolvlm2) |
| S15 | [Liquid AI LFM2.5-VL-1.6B model card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) |
| S16 | [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| S17 | [Google vision fine-tuning guide and hardware requirement](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora) |
| S18 | [llama.cpp multimodal runtime and mmproj documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md) |
| S19 | [Maintainer SmolVLM2-500M GGUF file listing](https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/tree/main) |
| S20 | [Unsloth Qwen3.5-0.8B GGUF file listing](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/main) |
| S21 | [Unsloth Qwen3.5-2B GGUF file listing](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/main) |
| S22 | [Official SmolVLM2-500M configuration](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/blob/main/config.json) |
| S23 | [Official Qwen3.5-2B configuration](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json) |
| S24 | [NVIDIA T4 product specification](https://www.nvidia.com/en-sg/data-center/tesla-t4/) |
| S25 | [FlashAttention hardware-support documentation](https://github.com/Dao-AILab/flash-attention/blob/main/README.md) |
| S26 | [OpenFake authors’ dataset](https://huggingface.co/datasets/ComplexDataLab/OpenFake) |
| S27 | [llama.cpp Android build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) |
| S28 | [Google LiteRT-LM overview](https://ai.google.dev/edge/litert-lm/overview) |
| S29 | [Android dumpsys memory documentation](https://developer.android.com/tools/dumpsys) |
| S30 | [Arm Android Qwen3-VL deployment guide](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/ai-portal-mobile-vision-language/3-run-model/) |
| R1 | [Pinned README](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/README.md) |
| R2 | [Pinned release_manifest.json](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/assets/constants/release_manifest.json) |
| R3 | [Pinned LiteRTInferenceEngine.kt](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/java/com/onndetect/inference/LiteRTInferenceEngine.kt) |
| R4 | [Pinned DeepThinkAlgorithmsTest.kt](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/test/java/com/onndetect/DeepThinkAlgorithmsTest.kt) |
| R5 | [Pinned saved LLM experiment results](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/ml/scripts/llm_signal_analysis_results.json) |
| R6 | [Pinned LLM experiment runner](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/ml/scripts/llm_signal_experiment.py) |
| R7 | [Pinned AndroidManifest.xml](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/AndroidManifest.xml) |
| R8 | [Pinned DetectionPipeline.kt](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/java/com/onndetect/domain/pipeline/DetectionPipeline.kt) |
| R9 | [Pinned VfmProbeAnalyzer.kt](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/java/com/onndetect/detection/visual/VfmProbeAnalyzer.kt) |
| R10 | [Pinned SigLIP LFS pointer and byte size](https://github.com/shubhamiitshk/Onn-Detect-/blob/3da11a3d13690a1d3c550410e797030517a8756b/android/app/src/main/assets/models/siglip_probe_int8.tflite) |
