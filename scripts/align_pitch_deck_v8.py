import os
import sys
from pptx import Presentation

sys.stdout.reconfigure(encoding='utf-8')

pptx_path = r'c:\Users\PLEXTEK\Desktop\Onn-Detect-\ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx'
prs = Presentation(pptx_path)

def set_para_text(para, text):
    if len(para.runs) > 0:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r.text = ""
    else:
        para.text = text

def set_run_pair(para, bold_prefix, rest):
    if len(para.runs) >= 2:
        para.runs[0].text = bold_prefix
        para.runs[1].text = rest
        for r in para.runs[2:]:
            r.text = ""
    elif len(para.runs) == 1:
        para.runs[0].text = bold_prefix + rest
    else:
        para.text = bold_prefix + rest

# --- SLIDE 1 ---
# Title & Hook
s1 = prs.slides[0]
set_para_text(s1.shapes[1].text_frame.paragraphs[2], "On-device deepfake, voice-clone, and KYC fraud defense in under 450 ms.")
set_para_text(s1.shapes[2].text_frame.paragraphs[0], "Release: onn-20261023-v8  |  Stage: Pre-Seed (TRL 6-7)  |  IIT Kharagpur")

# --- SLIDE 2 ---
# The Core Problem
s2 = prs.slides[1]
# Col 1: Voice & Receipt Fraud
set_para_text(s2.shapes[2].text_frame.paragraphs[0], "Voice & Receipt Fraud")
set_run_pair(s2.shapes[2].text_frame.paragraphs[1], "Voice-Clone Scams: ", "Forwarded AI voice notes impersonate family members to extort instant UPI transfers.")
set_run_pair(s2.shapes[2].text_frame.paragraphs[2], "Forged Screenshots: ", "AI-edited payment receipts and fake invoices defraud merchants and insurers daily.")
set_run_pair(s2.shapes[2].text_frame.paragraphs[3], "No Local Verification: ", "Consumers lack offline identity + authenticity verification for shared media.")

# Col 2: Video-KYC & HAL Spoofing
set_para_text(s2.shapes[3].text_frame.paragraphs[0], "Video-KYC & HAL Spoofing")
set_run_pair(s2.shapes[3].text_frame.paragraphs[1], "Virtual-Camera Hooks: ", "Attackers inject pre-rendered deepfakes into bank Video-KYC (V-CIP) sessions.")
set_run_pair(s2.shapes[3].text_frame.paragraphs[2], "Lip-Sync Synthesis: ", "Audio-driven talking-head fakes bypass single-frame face liveness checks.")
set_run_pair(s2.shapes[3].text_frame.paragraphs[3], "Compression Collapse: ", "Academic CNNs fail after WhatsApp JPEG Q85 or VoIP Opus re-encoding.")

# Col 3: Cloud Unit Economics
set_para_text(s2.shapes[4].text_frame.paragraphs[0], "Cloud Unit Economics")
set_run_pair(s2.shapes[4].text_frame.paragraphs[1], "Privacy Violations: ", "Uploading voiceprints and biometric KYC video violates DPDP Act and GDPR.")
set_run_pair(s2.shapes[4].text_frame.paragraphs[2], "Unsustainable Cost: ", "Cloud GPU APIs cost $0.02-$0.10 per scan, blocking high-volume screening.")
set_run_pair(s2.shapes[4].text_frame.paragraphs[3], "Unverified Provenance: ", "Stripped or forged C2PA metadata cannot be trusted without offline crypto checks.")

# --- SLIDE 3 ---
# The ONN-DETECT Solution
s3 = prs.slides[2]
set_run_pair(s3.shapes[3].text_frame.paragraphs[0], "Zero-Network Sandbox: ", "Strips INTERNET permission at the OS manifest level; voiceprints and KYC media never leave the phone.")
set_run_pair(s3.shapes[3].text_frame.paragraphs[1], "5-Layer Edge Cascade: ", "Unifies INT8 SigLIP-B/16 (94.3 MB), 20-ch SRM/NPR/RGB CNN, LCNN-MFM audio, BioLip, and C2PA.")
set_run_pair(s3.shapes[3].text_frame.paragraphs[2], "Play-Compliant B2B2C Suite: ", "Runs in 282 ms (4 threads) across our consumer Forward Guard + Voice Vault app and enterprise KYC SDK.")
set_para_text(s3.shapes[4].text_frame.paragraphs[0], "ONN-DETECT  |  Zero-Cloud Edge Architecture (Release onn-20261023-v8)  |  Slide 3")

# --- SLIDE 4 ---
# Consumer & Forensic Capabilities
s4 = prs.slides[3]
set_para_text(s4.shapes[1].text_frame.paragraphs[0], "CONSUMER & FORENSIC CAPABILITIES")
set_para_text(s4.shapes[1].text_frame.paragraphs[1], "Play-Compliant Consumer & Forensic Suite Verified by 220 JVM Tests")

# Col 1: Family Safe-Voice Vault
set_para_text(s4.shapes[3].text_frame.paragraphs[0], "Family Safe-Voice Vault")
set_run_pair(s4.shapes[3].text_frame.paragraphs[1], "Dual-Gate Verdict: ", "Gates on LCNN-MFM (<0.375) + 3s/1.5s window voiceprints.")
set_run_pair(s4.shapes[3].text_frame.paragraphs[2], "AES-256-GCM Sealed: ", "Seals voiceprints in Room with Keystore AAD binding.")
set_run_pair(s4.shapes[3].text_frame.paragraphs[3], "Clone Alert Path: ", "Flags synthetic audio matching a relative (CAM++ next).")

# Col 2: Instant Forward Guard
set_para_text(s4.shapes[5].text_frame.paragraphs[0], "Instant Forward Guard")
set_run_pair(s4.shapes[5].text_frame.paragraphs[1], "1-Tap Share Target: ", "Handles ACTION_SEND & SEND_MULTIPLE for WhatsApp media.")
set_run_pair(s4.shapes[5].text_frame.paragraphs[2], "32x32 ROI Heatmap: ", "ResidualHeatmapExtractor pinpoints local tamper energy.")
set_run_pair(s4.shapes[5].text_frame.paragraphs[3], "ELA Document Guard: ", "DocumentForgeryAnalyzer catches spliced UPI receipts.")

# Col 3: Offline Provenance
set_para_text(s4.shapes[7].text_frame.paragraphs[0], "Offline Provenance")
set_run_pair(s4.shapes[7].text_frame.paragraphs[1], "32 Production Roots: ", "Verifies official C2PA anchors (Adobe, Google, DigiCert).")
set_run_pair(s4.shapes[7].text_frame.paragraphs[2], "Full Container Walk: ", "Reassembles JPEG APP11 (En, Z) & ISO-BMFF JUMBF boxes.")
set_run_pair(s4.shapes[7].text_frame.paragraphs[3], "Fail-Closed Crypto: ", "Requires chain + COSE_Sign1 + c2pa.hash.data match.")
set_para_text(s4.shapes[8].text_frame.paragraphs[0], "ONN-DETECT  |  100% Google Play Policy Compliant (Zero Accessibility RPA)  |  Slide 4")

# --- SLIDE 5 ---
# Enterprise B2B Monetization
s5 = prs.slides[4]
set_run_pair(s5.shapes[3].text_frame.paragraphs[0], "V-CIP KYC Injection Shield: ", "VirtualCameraDetector audits Camera2 HAL hardware levels, silicon sensor bounds, and Xposed hooks.")
set_run_pair(s5.shapes[3].text_frame.paragraphs[1], "Hardware Forensic Receipts: ", "StrongBoxReceiptSigner signs RFC 8785 canonical JSON + media SHA-256 via StrongBox/TEE ECDSA P-256.")
set_run_pair(s5.shapes[3].text_frame.paragraphs[2], "BioLip 478-Pt Mesh Defense: ", "MediaPipe FaceLandmarker extracts 64 perioral points (inter-ocular normalized) to catch lip-sync fakes.")

# --- SLIDE 6 ---
# Core Technological Innovation
s6 = prs.slides[5]
# Base-Margin Cascade
set_run_pair(s6.shapes[2].text_frame.paragraphs[1], "Continuous Tail Order: ", "Injects INT8 SigLIP-B/16 logit as XGBoost base_margin to prevent tree step-ties.")
set_run_pair(s6.shapes[2].text_frame.paragraphs[2], "20-Ch SRM + NPR Stem: ", "Combines 12 locked 5x5 SRM high-pass filters, 4 NPR kernels, and 4 RGB channels.")
set_run_pair(s6.shapes[2].text_frame.paragraphs[3], "Selective Arbitration: ", "Escalation band maximizes selective accuracy under a >=85% classified-rate floor.")

# ViT-to-LiteRT Compiler
set_run_pair(s6.shapes[3].text_frame.paragraphs[1], "1.97e-06 FP32 Parity: ", "Custom channels-last Keras mirror bypasses broken onnx2tf ViT shape inference.")
set_run_pair(s6.shapes[3].text_frame.paragraphs[2], "Static MAP Queries: ", "Pre-computes frozen attention-pooling queries, saving 589K weights at runtime.")
set_run_pair(s6.shapes[3].text_frame.paragraphs[3], "Zero-Heap APK mmap: ", "Maps the 94.3 MB INT8 tower into OS page cache (282 ms @ 4 threads, 0 MB heap).")

# Keys-Bicubic & Crypto
set_para_text(s6.shapes[4].text_frame.paragraphs[0], "Keys-Bicubic & Crypto")
set_run_pair(s6.shapes[4].text_frame.paragraphs[1], "Bit-Exact Resampler: ", "Custom two-pass Keys-cubic (a=-0.5) filter with 8-bit clip matches Pillow +/-1.")
set_run_pair(s6.shapes[4].text_frame.paragraphs[2], "DCPT Paired Loss: ", "Symmetric Bernoulli KL + penultimate cosine alignment hardens Q85 accuracy.")
set_run_pair(s6.shapes[4].text_frame.paragraphs[3], "Hardware Crypto Binding: ", "StrongBox ECDSA P-256 + 32-anchor C2PA COSE_Sign1 + AES-256-GCM AAD vault.")

# --- SLIDE 7 ---
# Measured Benchmark Validation
s7 = prs.slides[6]
set_run_pair(s7.shapes[3].text_frame.paragraphs[0], "7x Low-FPR Recall Gain: ", "Base-margin VFM fusion lifts image TPR@1%FPR from 4.0% to 28.0% and TPR@0.1%FPR to 25.0%.")
set_run_pair(s7.shapes[3].text_frame.paragraphs[1], "0.900 Laundered AUC: ", "Re-encoding test images to 512px JPEG Q85 raises v7 AUC to 0.900, proving social-feed resilience.")
set_run_pair(s7.shapes[3].text_frame.paragraphs[2], "0.875 Speech & 365 Gates: ", "LCNN-MFM reaches 59.5% TPR@1%FPR, backed by 220 JVM unit tests and 145 Python ML tests.")
set_para_text(s7.shapes[4].text_frame.paragraphs[0], "Source: ONN-DETECT Sealed Benchmark (3,600 Image / 68 Generators & 2,400 Audio Records)  |  Slide 7")

# --- SLIDE 8 ---
# Competitive Table
s8 = prs.slides[7]
table = s8.shapes[2].table
set_para_text(table.rows[0].cells[1].text_frame.paragraphs[0], "ONN-DETECT (v8)")
set_para_text(table.rows[4].cells[0].text_frame.paragraphs[0], "Hardware Receipts & HAL KYC")
set_para_text(table.rows[4].cells[1].text_frame.paragraphs[0], "StrongBox P-256 + Camera2 HAL")
set_para_text(table.rows[4].cells[2].text_frame.paragraphs[0], "No Device Attestation")
set_para_text(table.rows[4].cells[3].text_frame.paragraphs[0], "No Mobile Runtime")

set_para_text(table.rows[5].cells[0].text_frame.paragraphs[0], "Readiness & IP Status")
set_para_text(table.rows[5].cells[1].text_frame.paragraphs[0], "TRL 6-7 | Trade Secret + Patent")
set_para_text(table.rows[5].cells[2].text_frame.paragraphs[0], "Commercial Cloud SaaS")
set_para_text(table.rows[5].cells[3].text_frame.paragraphs[0], "TRL 2-3 Lab Papers")

# --- SLIDE 9 ---
# Market Size & Commercial Model
s9 = prs.slides[8]
set_run_pair(s9.shapes[3].text_frame.paragraphs[0], "Global TAM (Rs 62,000 Cr): ", "$7.4B global deepfake defense, Video-KYC anti-spoofing, and mobile voice-fraud market.")
set_run_pair(s9.shapes[3].text_frame.paragraphs[1], "B2B SDK (Rs 52.5 Cr SOM): ", "35 fintech, NBFC, and insurance clients at Rs 1.5 Cr avg ACV (Rs 20-Rs 40 per KYC/claim).")
set_run_pair(s9.shapes[3].text_frame.paragraphs[2], "B2C & Traction (Rs 32.5 Cr SOM): ", "1.1M users at Rs 299/yr (Safe-Voice Vault & Forward Guard); TRL 6-7 APK ready for pilots.")

# --- SLIDE 10 ---
# Funding Ask & Runway
s10 = prs.slides[9]
set_run_pair(s10.shapes[4].text_frame.paragraphs[1], "Runway & P1-P3 Milestones: ", "Rs 2.08L/mo burn (12 mos); ships CAM++ speaker encoder, 400-video calibration, & device gates.")
set_run_pair(s10.shapes[4].text_frame.paragraphs[2], "CapEx (35% / Rs 8.75L): ", "GPU compute for B-Free twins & 400-video split (Rs 5.25L) + Android NPU test matrix (Rs 3.50L).")
set_run_pair(s10.shapes[4].text_frame.paragraphs[3], "OpEx (65% / Rs 16.25L) & Support: ", "Eng stipends (Rs 9.5L), pilots (Rs 3.5L), patents/audit (Rs 3.25L) + KGP Forge mentorship.")

# Save updated presentation
targets = [
    r'c:\Users\PLEXTEK\Desktop\Onn-Detect-\ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx',
    r'c:\Users\PLEXTEK\Desktop\Onn-Detect-\docs\ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx',
    r'C:\Users\PLEXTEK\.gemini\antigravity\brain\ab162997-2a8d-4152-a5b1-906b2f5331ee\ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx',
]
for t in targets:
    if os.path.exists(os.path.dirname(t)):
        prs.save(t)
        print(f"Updated: {t}")
