import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


INPUT_SCALE = 64.0  # quantization scale for normalized inputs


def get_ml_dir(default_dir: Path) -> Path:
    """Resolve ML directory, allowing override via ML_DIR env var."""
    env_dir = os.environ.get("ML_DIR")
    return Path(env_dir) if env_dir is not None else default_dir


def load_scaler(ml_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.load(ml_dir / "scaler_mean.npy")
    scale = np.load(ml_dir / "scaler_scale.npy")
    return mean, scale


def load_quantized_weights(ml_dir: Path):
    W1_q = np.load(ml_dir / "W1_int8.npy")
    W2_q = np.load(ml_dir / "W2_int8.npy")
    b1_q = np.load(ml_dir / "b1_int8.npy")
    b2_q = np.load(ml_dir / "b2_int8.npy")
    scales = np.load(ml_dir / "scale_factors.npy", allow_pickle=True).item()

    W1_scale = float(scales["W1_scale"])
    W2_scale = float(scales["W2_scale"])
    b1_scale = float(scales["b1_scale"])
    b2_scale = float(scales["b2_scale"])

    return W1_q, W2_q, b1_q, b2_q, W1_scale, W2_scale, b1_scale, b2_scale


def load_float_model(ml_dir: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(ml_dir / "model.h5")


def vectorized_int8_inference(
    norm_inputs: np.ndarray,
    W1_q: np.ndarray,
    W2_q: np.ndarray,
    b1_q: np.ndarray,
    b2_q: np.ndarray,
    W1_scale: float,
    W2_scale: float,
    b1_scale: float,
    b2_scale: float,
    input_scale: float = INPUT_SCALE,
):
    """
    Fully vectorized INT8 inference over all samples.

    norm_inputs: (N, D_in) normalized float32 inputs.
    Returns:
      x_q      : (N, D_in) int8 quantized inputs
      probs    : (N, num_classes) float64 probabilities
      preds    : (N,) predicted class indices
    """
    # Quantize inputs to INT8
    x_q = np.clip(
        np.round(norm_inputs * input_scale), -128, 127
    ).astype(np.int8)

    # Convert to float for matmuls
    x_f = x_q.astype(np.float32) / input_scale

    # Dequantize weights/biases once
    W1_f = W1_q.astype(np.float32) / W1_scale
    b1_f = b1_q.astype(np.float32) / b1_scale
    W2_f = W2_q.astype(np.float32) / W2_scale
    b2_f = b2_q.astype(np.float32) / b2_scale

    # Layer 1: ReLU(W1 * x + b1)
    h = np.maximum(0.0, x_f @ W1_f + b1_f)

    # Layer 2: logits = W2 * h + b2
    logits = h @ W2_f + b2_f

    # Softmax in a numerically stable, vectorized way
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=1, keepdims=True)

    preds = probs.argmax(axis=1)
    return x_q, probs, preds


def generate_inputs(n_each: int, seed: int) -> np.ndarray:
    """
    Generate raw sensor inputs:
      - ~n_each safe
      - ~n_each warning
      - ~n_each danger
    """
    np.random.seed(seed)

    # Safe inputs - all sensors in normal range
    safe_inputs = np.column_stack(
        [
            np.random.uniform(0, 40, n_each),   # PM25 well below 55
            np.random.uniform(0, 200, n_each),  # VOC well below 300
            np.random.uniform(20, 30, n_each),  # HeatIdx well below 35
            np.random.uniform(60, 90, n_each),  # HR well below 100
            np.random.uniform(96, 100, n_each), # SpO2 well above 95
        ]
    )

    # Warning inputs - at least one sensor in warning range
    warning_inputs = np.column_stack(
        [
            np.random.uniform(60, 140, n_each),  # PM25 clearly in warning
            np.random.uniform(0, 200, n_each),   # VOC safe
            np.random.uniform(20, 30, n_each),   # HeatIdx safe
            np.random.uniform(60, 90, n_each),   # HR safe
            np.random.uniform(96, 100, n_each),  # SpO2 safe
        ]
    )

    # Danger inputs - at least one sensor in danger range
    danger_inputs = np.column_stack(
        [
            np.random.uniform(151, 500, n_each),  # PM25 in danger range
            np.random.uniform(0, 299, n_each),    # VOC normal
            np.random.uniform(20, 34, n_each),    # HeatIdx normal
            np.random.uniform(40, 99, n_each),    # HR normal
            np.random.uniform(95, 100, n_each),   # SpO2 normal
        ]
    )

    raw_inputs = np.vstack([safe_inputs, warning_inputs, danger_inputs])
    return raw_inputs


def build_results_dataframe(
    raw_inputs: np.ndarray,
    x_q: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    float_preds: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            # Raw sensor values
            "pm25": np.round(raw_inputs[:, 0], 4),
            "voc": np.round(raw_inputs[:, 1], 4),
            "heatidx": np.round(raw_inputs[:, 2], 4),
            "hr": np.round(raw_inputs[:, 3], 4),
            "spo2": np.round(raw_inputs[:, 4], 4),
            # INT8 quantized inputs (what Verilog receives)
            "in_q0": x_q[:, 0].astype(int),
            "in_q1": x_q[:, 1].astype(int),
            "in_q2": x_q[:, 2].astype(int),
            "in_q3": x_q[:, 3].astype(int),
            "in_q4": x_q[:, 4].astype(int),
            # Output probabilities
            "out_safe": np.round(probs[:, 0].astype(float), 6),
            "out_warning": np.round(probs[:, 1].astype(float), 6),
            "out_danger": np.round(probs[:, 2].astype(float), 6),
            # Final classifications
            "label": preds.astype(int),
            # Float model predictions for comparison
            "float_label": float_preds.astype(int),
        }
    )
    return df


def export_verilog_testbench(df: pd.DataFrame, output_path: Path):
    """
    Export test vectors in Verilog $readmemh format.
    
    Format for $readmemh:
    - Hexadecimal values only (no 0x prefix)
    - One value per line
    - Optional comments start with //
    - Can use @address format (not needed here)
    """
    with open(output_path, 'w') as f:
        f.write("// Golden test vectors for Verilog testbench\n")
        f.write("// Generated from INT8 quantization validation\n")
        f.write("// Format: {in_q0[7:0], in_q1[7:0], in_q2[7:0], in_q3[7:0], in_q4[7:0], expected_label[1:0]}\n")
        f.write("// Total bits: 42 bits (5*8 + 2)\n\n")
        
        for idx, row in df.iterrows():
            # Pack quantized inputs into a single value
            packed = 0
            for i in range(5):
                # Convert signed 8-bit to unsigned for packing
                val = row[f'in_q{i}'] & 0xFF  # Mask to 8 bits
                packed = (packed << 8) | val
            
            # Add expected label (2 bits)
            packed = (packed << 2) | (row['label'] & 0x3)
            
            # Format as hex with consistent width (11 hex chars for 42 bits)
            hex_str = f"{packed:011X}"  # 11 hex characters, uppercase, no 0x prefix
            
            # Add comment with raw values for debugging
            f.write(f"{hex_str}  // {idx:3d}: PM2.5={row['pm25']:6.1f}, "
                   f"VOC={row['voc']:6.1f}, HeatIdx={row['heatidx']:5.1f}, "
                   f"HR={row['hr']:5.1f}, SpO2={row['spo2']:5.1f} -> "
                   f"{['Safe', 'Warning', 'Danger'][row['label']]}\n")
    
    print(f"Verilog test vectors written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate golden vectors for INT8 vs float model."
    )
    parser.add_argument(
        "--n-each",
        type=int,
        default=334,
        help="Number of samples per class bucket (safe/warning/danger). "
        "Total samples = 3 * n_each. Default: 334.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for input generation (default: 123).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_vectors.csv",
        help="Output CSV path (default: golden_vectors.csv in script directory).",
    )
    parser.add_argument(
        "--ml-dir",
        type=Path,
        default=None,
        help="Directory containing model/weights/scaler files. "
        "Defaults to ML_DIR env var if set, otherwise script directory.",
    )
    parser.add_argument(
        "--verilog-out",
        type=Path,
        default=None,
        help="Optional path for Verilog testbench hex output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    base_ml_dir = args.ml_dir if args.ml_dir is not None else get_ml_dir(script_dir)

    mean, scale = load_scaler(base_ml_dir)
    (
        W1_q,
        W2_q,
        b1_q,
        b2_q,
        W1_scale,
        W2_scale,
        b1_scale,
        b2_scale,
    ) = load_quantized_weights(base_ml_dir)
    model = load_float_model(base_ml_dir)

    raw_inputs = generate_inputs(args.n_each, args.seed)
    norm_inputs = (raw_inputs - mean) / scale

    # Float32 model predictions (already vectorized by Keras)
    float_probs = model.predict(norm_inputs.astype(np.float32), verbose=0)
    float_preds = np.argmax(float_probs, axis=1)

    # Vectorized INT8 inference
    x_q, probs, preds = vectorized_int8_inference(
        norm_inputs=norm_inputs.astype(np.float32),
        W1_q=W1_q,
        W2_q=W2_q,
        b1_q=b1_q,
        b2_q=b2_q,
        W1_scale=W1_scale,
        W2_scale=W2_scale,
        b1_scale=b1_scale,
        b2_scale=b2_scale,
    )

    df = build_results_dataframe(raw_inputs, x_q, probs, preds, float_preds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    matches = int((preds == float_preds).sum())
    N = len(raw_inputs)

    print(f"Generated {N} golden vectors")
    print(f"INT8 vs Float32 agreement: {matches}/{N}")
    print("\nClass distribution:")
    print(f"  Safe:    {(df.label == 0).sum()}")
    print(f"  Warning: {(df.label == 1).sum()}")
    print(f"  Danger:  {(df.label == 2).sum()}")
    print("\nFirst 3 vectors:")
    print(df[["pm25", "voc", "heatidx", "hr", "spo2", "in_q0", "label"]]
          .head(3)
          .to_string())
    print(f"\ngolden_vectors.csv saved to {args.output}")
    
    # Export Verilog testbench if requested
    if args.verilog_out:
        export_verilog_testbench(df, args.verilog_out)


if __name__ == "__main__":
    main()
# Updated for Week 1 & 2

# Updated for Week 1 & 2 Project
