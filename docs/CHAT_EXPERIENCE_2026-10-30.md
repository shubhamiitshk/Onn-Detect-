# Chat experience revision — 30 October 2026

The Android app now has a cleaner, private chat interface with direct attachment handling and incremental responses.

## Delivered

- A multiline composer with send, stop, retry, copy, new conversation, a running circle, typing cursor, and scroll-to-latest behavior.
- Up to four attachments per message: supported images, video, audio, PDF, plain text, Markdown, CSV and JSON. Files have previews and removal controls before sending.
- Follow-up context from recent messages and the four most recent indexed sources. A fresh chat does not silently open a previous scan.
- Calmer glass surfaces, restrained highlights, readable contrast, a keyboard-aware dock, and larger-text interaction checks.
- Bounded imports and document parsing. Image/audio explanations consume shared analysis outcomes; missing or unvalidated evidence remains inconclusive. Video frames remain diagnostic until recording-level calibration exists.
- Source-labelled retrieval and citations. PDFs expose selectable text on Android 15/API 35 and later; inspection is limited to the first three pages. Scanned PDFs need OCR, which is not implemented.
- Genuine native token callbacks for an installed compatible, verified local model pack, with incremental UTF-8 handling and cooperative cancellation. The Qwen3.5 response prefix follows the local model's non-thinking template.
- Built-in evidence summaries when no usable model pack is installed, labelled as such. These summaries animate progressively; they are distinct from native model generation.
- Native context ownership is synchronized, and chat unloads model residency before releasing its exclusive inference lease.

## Product boundary

The chat interaction resembles familiar conversational assistants; the app is not connected to the ChatGPT service. General-purpose generated replies require a compatible verified local language-model pack. Attachment analysis and built-in evidence replies work without a network connection. Conversation state is retained by the ViewModel during the current app session; this change does not add a persistent chat-history database.

The residual/semantic/fusion image architecture remains a suitable starting point. This UI revision does not retrain or establish production accuracy for its existing models, repair every provenance issue from the audit, or provide independently validated device detection performance. The October 29 project audit remains the completion backlog for those concerns.

## Verification

- **413 JVM tests passed**, with zero failures, errors or skips.
- **11 emulator tests passed** on API 36: seven Compose interaction/preview cases, two real ViewModel/provider/document-flow cases, and two genuine Qwen GGUF streaming/cancellation cases. The real model weights were explicitly deployed to the test emulator, so the native cases executed rather than skipping.
- The minified release build succeeded. APK verification passed for merged network permissions, model/artifact contracts and native 16 KiB alignment. Source consistency and documentation checks passed.
- Android lint reported zero errors and 81 warnings. These are implementation/build checks; no detector accuracy or physical-phone performance claim follows from them.
- [Verification summary](test_artifacts/chat_experience_2026-10-30_summary.json), [empty chat preview](../outputs/chat-ui/ONN-Chat-Preview/chat-empty.png), [streaming chat preview](../outputs/chat-ui/ONN-Chat-Preview/chat-streaming.png).

The preview images are real Android Compose renders with synthetic design-fixture messages, not accuracy examples. General conversational output still depends on an installed compatible verified local model pack.
