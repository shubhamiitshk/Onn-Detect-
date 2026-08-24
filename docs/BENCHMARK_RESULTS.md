# Benchmark Results

Source of truth: `ml/evaluation_results.json`, produced by
`ml/scripts/run_full_pipeline.py` on real held-out data. Re-run it anytime the
checkpoints or datasets change.

| Stage | Eval split | n | Metric | Value |
|---|---|---|---|---|
| SRMForensicCNN (visual) | genimage/val, native 32x32 input | 4000 | AUC / Acc / EER | 0.956 / 0.889 / 0.110 |
| LCNN-MFM (audio) | asvspoof2019/test | 280 | EER / AUC | 0.050 / 0.976 |
| XGBoost meta fusion | held-out feature split | 1000 | AUC / Acc | 0.962 / 0.893 |

## Caveats (read before quoting)

- The visual model is trained and evaluated at **native 32x32 resolution**.
  Upscaling inputs destroys the high-frequency statistics it depends on
  (see `ml/scripts/retrain_srm_native.py` for the failed 224x224 ablation).
- Meta-classifier features 6-9 (exif/c2pa/lip-sync/jitter) currently carry
  neutral constants on both platforms; until retrained with real per-file
  labels they contribute no discriminative signal, so treat the meta AUC as
  an upper bound driven mostly by features 0-5.
- The transpiled Kotlin classifier is verified against
  `booster.predict()` to a max deviation < 1e-4 at export time
  (`train_xgboost_meta.py::validate_export`).

## Reproduce

```bash
cd ml && make eval          # writes ml/evaluation_results.json
python ../scripts/check_android_consistency.py
```
