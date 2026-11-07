# Qwen import and mobile chat UI correction — 30 October 2026

## Failure reproduced from the supplied screenshot

The prior importer accepted at most four ZIP entries and required a flat, app-specific conversation manifest. The existing `outputs/qwen35-0.8b-v1.zip` contains five entries: language GGUF, OCR projector GGUF, forensic policy, publisher key, and signed forensic manifest. It triggers the exact **“Duplicate or excessive files in the chat pack”** error shown in the screenshot, despite having no duplicate filenames. The error combined two different conditions and gave no usable recovery action.

This is an archive compatibility failure before inference, not evidence of insufficient phone RAM or failed Qwen generation. The user subsequently confirmed selecting this exact filename. This is the concrete reproduction; other Qwen ZIPs can have different contents.

## Import behavior

- Choose a Qwen3.5 language `.gguf` directly, an app conversation ZIP, or a ZIP containing one language GGUF and an optional vision GGUF.
- Directory entries, a common wrapper folder, and ordinary README/license/config sidecars no longer cause rejection. Only model files and the conversation contract are installed.
- The legacy five-file archive supplies its actual language weights for normal conversation. Its forensic policy/head is not required or activated for chat.
- Language and projector roles come from bounded GGUF metadata. Multiple language files, multiple projectors, duplicate paths/basenames, unsafe paths, damaged ZIPs and oversized content receive separate actionable errors.
- A supplied `chat_model.json` remains authoritative. Invalid declarations, malformed projector types or mismatched hashes fail; the app does not generate a replacement contract to conceal invalid supplied metadata.
- A manifestless import receives a local contract containing the exact inspected model identity, file sizes and SHA-256 hashes. This establishes local byte identity, not publisher authentication or forensic validation.
- Extraction remains bounded: 256 entries, at most eight candidate GGUF files, 3 GiB total expanded content including ignored files, small sidecar limits, current free-storage reserve, and cancellation. Activation stages separately and retains the previous model if import fails.

The legacy projector is an OCR fine-tune previously shown to fail a neutral-filename colour test with the base language model. It stays hash-checked but **does not enable vision**. The screen explains that text chat works and a matching base vision pack is needed for image/video questions. The exact base language/projector pair already exercised by real pixel tests remains enabled. Other unverified vision pairings are labelled unavailable rather than silently declared working.

## Interface correction

Model setup now leads with one model card and **Choose model file**. It shows the selected filename, real import stage, a working Cancel action, and a direct **Start chatting** action after installation. Failed imports offer **Choose another file**. Device information is one compact line. Technical descriptions, hashes, research controls and removal controls sit under expandable details.

Chat now uses a compact model-aware toolbar, small Chat/Media/Reports controls, a smaller centered welcome state, compact attachment chips, and one streaming indicator. Source details scroll with messages instead of consuming a pinned block above the conversation. The composer retains 48 dp controls and keyboard-aware placement. The model setup status is actionable without the old large explanation card.

## Phone setup

1. Install the corrected APK over the previous app; clearing app data is unnecessary.
2. Open the model selector in chat and choose **Choose model file**.
3. Select the original `qwen35-0.8b-v1.zip` for text chat, or `qwen35-0.8b-base-chat.zip` for the tested text-and-vision pairing. A direct supported language GGUF also enables text chat.
4. When installation finishes, choose **Start chatting**. Vision capability is stated separately from text capability.

## Verification

The final verification record is saved as `docs/test_artifacts/chat_import_ui_fix_2026-10-30_summary.json`, with connected results alongside it. New checks cover the actual five-file archive through the real Android provider/importer and genuine native language generation; failed replacement preserving the installed model; actual import-job cancellation; expanded archive unit cases; model setup states; chat with the software keyboard visible; and large text. Rendered previews are UI fixtures, not accuracy evidence.

All **432 JVM tests passed**. The first connected run passed **24 functional checks**, including the confirmed five-file ZIP and real Qwen generation, model setup states, existing visual/report chat and cancellation. Its additional keyboard case used a Compose input harness that intercepts native IME sessions and was replaced by a real Android input test. Subsequent native attempts were obstructed by an emulator SystemUI ANR, recorded in `outputs/import-ui-fix/keyboard-check.png`. After closing that dialog and reducing only the disposable emulator's framebuffer, the actual keyboard test passed: real typed text reaches the input, the Send button remains above the keyboard, and the app's bottom dock disappears.

The original 24 passing cases ran at 1080×2400 / density 420. The final keyboard case ran at 720×1600 / density 280, preserving the same logical phone dimensions. These are **25 passing Android checks across the original and focused runs**, not a claim that the initial 25-case run was failure-free. Raw initial and focused XML results are retained. The emulator was closed after verification.

The minified release build completed. APK signature verification, asset/model integrity, absence of network permissions and 16 KiB native alignment passed. Android contract and documentation checks passed. Lint reported zero errors and **74 warnings**. The checked installer is `outputs/apk/ONN-Detect-chat-import-ui-fix-2026-10-30.apk`. Preview files are in `outputs/import-ui-fix/`, including the native keyboard capture under `ONN-Import-UI-Preview/`.

This revision does not establish independent physical-phone detector accuracy, universal latency or support for Safetensors archives. Existing detector abstention and validation boundaries remain in force.
