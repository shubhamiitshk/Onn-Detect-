import tensorflow as tf
import numpy as np
import os


def quantize_keras_model(keras_model, representative_dataset_gen, output_path: str, allow_dynamic_range: bool = False):
    """
    Quantizes a Keras model straight to INT8 TFLite with a REAL-data
    representative dataset. Dynamic range fallback requires an explicit
    diagnostic opt-in; a release conversion fails on unsupported INT8 ops.
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
        if not allow_dynamic_range:
            raise RuntimeError("Full INT8 conversion failed; no silent release fallback") from e
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


def quantize_saved_model(fp32_tflite_path: str, representative_dataset_gen, output_path: str, allow_dynamic_range: bool = False):
    """
    Quantizes a SavedModel DIRECTORY; TFLite flatbuffers cannot be reconverted.
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
        if not allow_dynamic_range:
            raise RuntimeError("Full INT8 conversion failed; no silent release fallback") from e
        print(f"Full-int8 conversion failed ({e}); falling back to dynamic range.")
        converter.representative_dataset = None
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        quantized = converter.convert()
        kind = "dynamic-range"

    with open(output_path, 'wb') as f:
        f.write(quantized)

    fp32_size = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(fp32_tflite_path) for f in files) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{kind}] {fp32_size:.2f} MB -> {int8_size:.2f} MB "
          f"({(1 - int8_size / fp32_size) * 100:.1f}% smaller)")
    return kind


def visual_calibration_dataset(image_paths, limit=200, input_size=32):
    """Yield versioned decode + square resize, NHWC float RGB in [0,1]."""
    from PIL import Image
    from data.image_frontend import decode_image

    def generator():
        count = 0
        for p in image_paths:
            if count >= limit:
                break
            img = decode_image(p).image.resize((input_size, input_size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            yield [arr[np.newaxis]]
            count += 1

    return generator


def semantic_calibration_dataset(image_paths, limit=200, input_size=224):
    """Yield Engine C ("Lite C") appearance tensors from REAL images.

    The Engine C frontend is bicubic + CLIP mean/std normalization, which is
    deliberately DIFFERENT from Engine A's bilinear + [0,1] scale. Sharing a
    resampling artifact between the two engines would make the fusion measure
    the same low-level cue twice.
    """
    from data.image_frontend import semantic_image_tensor, decode_image

    def generator():
        count = 0
        for p in image_paths:
            if count >= limit:
                break
            tensor = semantic_image_tensor(decode_image(p).image, input_size)
            arr = np.transpose(tensor.numpy(), (1, 2, 0))[np.newaxis]
            yield [arr.astype(np.float32)]
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
