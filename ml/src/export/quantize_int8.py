import tensorflow as tf
import numpy as np
import os


def quantize_keras_model(keras_model, representative_dataset_gen, output_path: str):
    """
    Quantizes a Keras model straight to INT8 TFLite with a REAL-data
    representative dataset. Falls back to dynamic range quantization when
    full-int8 conversion fails.
    """
    print(f"Quantizing {keras_model.name} to INT8...")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Full integer ops; float in/out so Android needs no manual scaling.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    try:
        quantized = converter.convert()
        kind = "full-int8"
    except Exception as e:
        print(f"Full-int8 conversion failed ({e}); falling back to dynamic range.")
        converter.representative_dataset = None
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        quantized = converter.convert()
        kind = "dynamic-range"

    with open(output_path, 'wb') as f:
        f.write(quantized)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[{kind}] -> {output_path} ({size_kb:.0f} KB)")
    return kind


def quantize_tflite_model(fp32_tflite_path: str, representative_dataset_gen, output_path: str):
    """
    Quantizes a floating-point TFLite flatbuffer to INT8 using Post-Training
    Quantization (kept for SavedModel/flatbuffer inputs).
    """
    print(f"Quantizing {fp32_tflite_path} to INT8...")

    converter = tf.lite.TFLiteConverter.from_saved_model(fp32_tflite_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Full integer ops; float in/out so Android needs no manual scaling.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    try:
        quantized = converter.convert()
        kind = "full-int8"
    except Exception as e:
        print(f"Full-int8 conversion failed ({e}); falling back to dynamic range.")
        converter.representative_dataset = None
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        quantized = converter.convert()
        kind = "dynamic-range"

    with open(output_path, 'wb') as f:
        f.write(quantized)

    fp32_size = os.path.getsize(fp32_tflite_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{kind}] {fp32_size:.2f} MB -> {int8_size:.2f} MB "
          f"({(1 - int8_size / fp32_size) * 100:.1f}% smaller)")
    return kind


def visual_calibration_dataset(image_paths, limit=200):
    """Yields NHWC (1,32,32,3) float RGB in [0,1] from REAL images.

    Exactly matches training preprocessing: ToTensor == uint8/255, native res,
    no resize, no normalization.
    """
    from PIL import Image

    def generator():
        count = 0
        for p in image_paths:
            if count >= limit:
                break
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if arr.shape != (32, 32, 3):
                continue
            yield [arr[np.newaxis]]
            count += 1

    return generator


def audio_specs_calibration_dataset(specs_pt_layout, limit=50):
    """Yields NHWC (1,T,F,1) LFCC tensors from REAL spectrograms.

    specs_pt_layout: list of PyTorch-layout (1,1,T,F) arrays as produced by
    LFCCExtractor.extract().
    """
    def generator():
        count = 0
        for s in specs_pt_layout:
            if count >= limit:
                break
            arr = np.transpose(np.asarray(s, dtype=np.float32)[0], (1, 2, 0))
            yield [arr[np.newaxis]]
            count += 1

    return generator
