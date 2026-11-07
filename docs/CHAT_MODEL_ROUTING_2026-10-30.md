# Local model routing implementation — 30 October 2026

This revision implements the three flows in the handwritten specification: ordinary conversation without attachments, questions about supported media/files, and Q&A over saved forensic scan reports.

## Model choice

The conversation model uses the existing **Qwen3.5-0.8B Q4 GGUF**. Normal chat uses its language weights; visual questions additionally load a matching base-model projector. A new `ChatModelStore` imports this conversation pack independently of the experimental forensic head. This removes the old requirement that ordinary chat have a trained forensic classification head.

Gemma 2 2B is a text generator; EmbeddingGemma 2 creates retrieval vectors. They are different model roles. The existing Gemma files are cached features and a head, not the complete deployable backbone. The [research and asset inventory](MODEL_SELECTION_AND_ROUTING_2026-10-30.md) cites the primary sources and records this distinction. No new Gemma model is downloaded or claimed active.

## Concrete artifacts

- Conversation archive: `outputs/chat-models/qwen35-0.8b-base-chat.zip` (678,961,164 bytes).
- Language file: 563,036,064 bytes, SHA-256 `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- Base vision projector: 115,923,680 bytes, SHA-256 `5a6b15da0f483b9e320a5b86b787ad1831953dfaa706e2c5d91de830ea70d784`.
- Projector source: `mradermacher/Qwen3.5-0.8B-GGUF`, immutable revision `78fc4c891ff7dee2e8286c3f4cdc8a26f7544a07`. Exact URL, declared base/license and independent digest verification are saved in [source evidence](test_artifacts/base_projector_source_2026-10-30.json).
- Original weights remain in `downloaded_models/`. The conversation archive preserves the selected bytes and does not repurpose the incompatible staged forensic fixture.

The earlier `qwen35-0.8b-chat.zip` preserves an OCR-finetuned projector and is **not the recommended pack**. A neutral-filename red-image test returned blue with that pairing. Earlier filename-hinted colour checks were insufficient evidence of visual quality. The OCR publisher also declares a separate fine-tuned language-model pairing and a noncommercial license; its projector cannot be assumed interchangeable with the base language weights. Existing OCR files and the earlier pack are preserved for reproducibility.

Install a newly built APK, open **Local models**, choose **Import conversation pack**, and select that archive from the phone's files. A host workspace file is not automatically accessible to an installed Android app. The model card shows the imported files, capabilities and CPU status. Hash checking establishes byte identity, not model accuracy or trusted publisher provenance.

## Routing and boundaries

| User flow | Execution |
|---|---|
| Chat | Language model with recent conversation context; no image projector or detector is loaded just to answer a general question. |
| Media question | Bounded imports, source hashes and actual image pixels; video uses a labelled contact sheet of up to four sampled frames. PDF can supply a bounded rendered page plus supported selectable text. |
| Forensic question about media | The shared specialist detector produces measurements/status; the generator explains them and retains uncertainty. |
| Scan reports | Picker reads actual local Room records, including saved image/audio/video outcomes; report IDs are looked up directly rather than opened as content URIs. Original media is not present in history. |
| No usable conversation model | Clearly labelled built-in evidence replies remain available. They are not presented as a generative model. |

Normal Chat clears active evidence sources. Media and report chips retain a bounded recent context. Each generated answer records actual runtime model ID, language digest and whether pixels were supplied, permitting tests to verify execution rather than a decorative model badge.

The language model does not replace SRM/SigLIP/fusion image classification or LCNN speech measurements. Landmark models remain video diagnostics. Research checkpoints, archived duplicates and feature caches are not silently loaded as runtime detectors. Existing accuracy/provenance/release gaps in the project audit remain separate work.

## Reliability and resource handling

The model store normalizes supported ZIP wrapper folders and ordinary sidecars into a flat private installation. It also accepts direct language GGUF files. Supplied conversation manifests remain authoritative; manifestless imports receive a byte-bound local contract. Supported GGUF identity, hashes, file/archive limits, free-storage reserve, cancellation, atomic staging/activation and previous-installation recovery are enforced. Model import/removal and generation share ownership. A native inference lease evicts the heavy semantic detector before model loading. Each request unloads its native context before releasing that lease. See the subsequent [import/UI correction and exact five-file ZIP reproduction](CHAT_IMPORT_UI_FIX_2026-10-30.md).

Memory checks use current availability and a declared conservative working reserve. They are scheduling estimates, not measured active RAM or universal minimum-phone guarantees. Text requests omit the projector. The native runtime uses CPU threads bounded by available hardware; no GPU/NPU acceleration is claimed.

Actual generation streams UTF-8 tokens and supports cancellation. Cancellation polling is throttled to avoid expensive Java thread attachment on every numerical kernel. Native timings log counts/elapsed time, not conversation text. Explicitly short questions have a smaller token budget; input and output context remain bounded. Ordinary image/video questions prepare visual context without loading forensic classifiers. Questions requesting forensic assessment use the specialist path. Conversation vision uses a bounded 16–256 image-token range; the forensic backend retains its default metadata-driven input contract.

Cancellation is verified during token generation. Initial native model/projector loading and vision encoding can take longer to return to a cancellation check; this revision does not promise immediate interruption at every loading stage. Audio attachments provide bounded speech-detector evidence, not speech transcription. Video questions use sampled visual frames and do not transcribe their audio.

Debug builds retain symbols but optimize inference kernels with `-O2`, avoiding the previous unoptimized numerical execution. No fast-math flags are introduced, so missing-value and finite-number contracts are preserved.

## Validation scope

The final Android run passed **420 JVM tests** and **17 connected tests**, with no failures or skips. The five Python packaging tests also passed. Connected tests ran on a disposable Android 16/API 36 x86_64 emulator with 3 GB RAM and two virtual CPUs. The selected pack was imported through the actual provider/importer and model store.

The minified release build completed. Lint reported zero errors and 83 warnings; it is not warning-free. APK inspection passed asset/model integrity, absence of network permissions, and 16 KiB alignment for both packaged 64-bit ABIs. Android contract and documentation checks passed. The disposable emulator was closed after testing.

| Acceptance case | Observed result |
|---|---|
| Normal chat with no attachments | Actual Qwen model ID and language digest recorded; six plus seven answered 13; no pixels or evidence KB supplied. |
| Red and blue images, both named `photo.png` | Both colours answered correctly from actual pixels with the matching base projector. Expected colours are not supplied in filenames or prompt text. |
| Encoded AVC/MP4 named `clip.mp4` | Production video decoder samples frames; projector receives actual contact-sheet pixels; dominant blue answered correctly; authenticity score remains unavailable. |
| Saved audio report | INCONCLUSIVE retained despite an inconsistent legacy verdict field; 60-second media duration comes from evidence rather than the 150 ms processing time; no image pixels claimed. |
| Native cancellation | Genuine token output is interrupted and the exclusive model lease is released. |
| Interface and report picker | Composer, attachments, stop/retry, two-times text scaling, report selection/staging and normal-chat evidence clearing pass. |

Final artifact hashes, build/lint/package outcomes and raw connected-test XML are recorded in [the verification summary](test_artifacts/chat_model_routing_2026-10-30_summary.json) and [connected results](test_artifacts/chat_model_routing_2026-10-30_connected.xml). Rendered UI fixtures are saved in `outputs/chat-ui/ONN-Chat-Preview/`; their preview conversations are synthetic, not detector or model-quality evidence.

Test model files were deliberately deployed to a disposable emulator; production app storage and original model files are preserved. Synthetic colours, an encoded uniform-colour clip and saved report fixtures exercise routing, not general VLM quality or forensic accuracy. Physical-phone accuracy, peak memory, sustained thermals and latency remain unmeasured. Emulator timings cannot establish the screenshot's iPhone/OnePlus performance claims. The detector release still lacks independent device validation, so this is a local test build rather than a completed validated forensic release.
