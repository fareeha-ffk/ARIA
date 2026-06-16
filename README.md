# ARIA — Air-Quality Recognition and Inference Accelerator

Real-time air quality classifier running a 2-layer INT8 neural network entirely in hardware on the Tang Nano 20K FPGA (GW2AR-18). No cloud. No external processor.

![Modules](https://img.shields.io/badge/RTL%20Modules-6%2F6-green)
![Golden Vectors](https://img.shields.io/badge/Golden%20Vectors-99.0%25-brightgreen)
![SVA](https://img.shields.io/badge/SVA%20Violations-0-blue)
![Tests](https://img.shields.io/badge/Tests-1046%2F1059-yellow)


## Vision

ARIA is built around a two-wearable concept: one device worn close to the body capturing physiological signals, and one worn externally capturing environmental exposure. The goal is to determine in real time whether pollution, heat, or poor ventilation is contributing to physical symptoms — not just whether air quality is poor in isolation.

<img width="1402" height="1122" alt="ChatGPT Image Jun 17, 2026 at 01_40_15 AM" src="https://github.com/user-attachments/assets/eb08010b-9660-4e7b-a6ba-0cf23ba005df" />

Fusing both streams lets the system distinguish between high pollution with no physiological effect, high pollution with active health impact, and physiological spikes from exertion rather than exposure. The full system targets a BiLSTM ensemble on the FPGA for temporal pattern analysis. Fixed-point quantization keeps inference within the device's power and memory bounds.

## Current Implementation — Phase 1

This repository implements the FPGA inference core. Phase 1 classifies five fused sensor channels (PM2.5, VOC, Heat Index, HR, SpO2) using a 2-layer INT8 neural network in Verilog RTL on the GW2AR-18. Full two-wearable integration and BiLSTM temporal fusion are planned for Phase 2.

## System Architecture (End-to-End Flow)

ARIA is a multi-domain sensing and inference system composed of:

1. **Data Acquisition Layer**
   - Body-worn physiological sensors
   - External environmental sensors
   - Outputs fused 5-channel vector

2. **Preprocessing Layer**
   - Normalization + scaling (INT8 conversion-ready)
   - Sensor validity masking (validity_reg)

3. **Inference Layer (FPGA)**
   - 2-layer INT8 neural network
   - Hardwired MAC pipeline (Verilog RTL)
   - Outputs: Safe / Warning / Danger

4. **System Control Layer**
   - Power FSM (clock gating)
   - Output FSM (alerts / LEDs / buzzer)

5. **Application Layer**
   - Mobile reporting interface
   - Symptom labeling for future training loop


<img width="1672" height="941" alt="generated-image" src="https://github.com/user-attachments/assets/e9a841e8-229b-4a40-9c0f-dcc1f9b5ac66" />

## ML Pipeline (Training → Quantization → Deployment)

### 1. Dataset Generation
- 10,000 synthetic samples (Dhaka environmental model)
- 5 features:
  - PM2.5
  - VOC
  - Heat Index
  - Heart Rate proxy
  - SpO2 proxy
- Class imbalance handling:
  - Oversampled "Warning" class

### 2. Model Architecture
- Fully Connected Network:
  - Input: 5
  - Hidden: 16 (ReLU)
  - Output: 3 (Softmax)

- Variants:
  - Baseline model
  - Dropout (20%) robust model

### 3. Training Strategy
- Loss: Cross-entropy
- Optimizer: Adam
- Evaluation:
  - Clean accuracy
  - Sensor-failure robustness testing

### 4. Quantization Pipeline
- Post-training INT8 quantization (TFLite-style)
- Input scale: 64.0
- Fixed-point constraints:
  - Weights: [-127, 127]
  - Bias: clipped INT8 range

- Outcome:
  - 4× memory reduction
  - ~0% accuracy loss (on golden set)
 
  - \begin{table}[h]
\centering
\caption{Dataset Statistics (10,000 Synthetic Samples)}
\label{tab:dataset}
\begin{tabular}{lrrrr}
\textbf{Feature} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Std} \\
PM2.5 (μg/m³) & 0 & 500 & 247.1 & 143.8 \\
VOC (ppb) & 0 & 1000 & 504.5 & 289.3 \\
Heat Index (°C) & 20 & 50 & 35.0 & 8.6 \\
Heart Rate (bpm) & 40 & 180 & 109.8 & 40.5 \\
SpO2 (\%) & 70 & 100 & 92.5 & 4.3 \\
\end{tabular}
\end{table}

\textbf{Distribution:} Warning 66.7\%, Danger 33.3\%, Safe 0\% (clinically hazardous classes emphasized; Safe trivially identifiable).
 
## Hardware–Software Co-Design Mapping

| ML Component | RTL Implementation |
|--------------|--------------------|
| Dense Layer 1 | MAC accumulator array (INT8) |
| ReLU | signed clamp logic |
| Dense Layer 2 | pipelined MAC tree |
| Softmax (implicit) | argmax comparator |
| Dropout (training only) | removed in inference |
| Input scaling | fixed-point multiplier |

## Repository Structure

## Verification Strategy

ARIA uses a 3-layer verification approach:

### 1. Unit Testing (RTL Level)
- Module-wise testbenches
- UART / FIFO / FSM validation

### 2. Golden Vector Validation
- 1,002 precomputed ML inference vectors
- Python INT8 model vs RTL output
- Measured parity: 99.0%

### 3. Formal Verification (SVA)
- FIFO overflow protection
- FSM deadlock freedom
- Power-state correctness
- Sensor validity invariants

## Performance Summary

- FPGA frequency: 50 MHz
- Inference latency: ~960 ns
- Throughput: >1M samples/sec (simulated)
- Resource usage:
  - LUT: 2.4%
  - FF: 2.0%
  - DSP: 50%
- Golden vector match: 99.0%
- Quantization error: 0% (bit-exact on 100 samples subset)

- ## Key Innovations

- Dual-stream sensing (physiological + environmental fusion)
- Real-time FPGA INT8 neural inference (no CPU dependency)
- Sensor validity-aware inference (graceful degradation)
- Hardware-safe ML pipeline with golden-vector parity testing
- Power-aware FSM for wearable deployment

- ## Limitations

- Synthetic dataset only (no real-world calibration yet)
- No PM2.5 ground-truth sensor integration in current hardware
- Model currently non-temporal (no BiLSTM yet)
- Mobile app is reporting-only (no closed-loop adaptation)

## Future Work

- Integration of PMS5003 / BME688 real sensors
- BiLSTM temporal fusion on FPGA
- On-device continual learning from user feedback
- Geo-tagged exposure heatmaps
- Personalized risk scoring per user
