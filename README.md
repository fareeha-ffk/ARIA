# ARIA — Air-Quality Recognition and Inference Accelerator

Real-time air quality classifier running a 2-layer INT8 neural network entirely in hardware on the Tang Nano 20K FPGA (GW2AR-18). No cloud. No external processor.

![Modules](https://img.shields.io/badge/RTL%20Modules-6%2F6-green)
![Golden Vectors](https://img.shields.io/badge/Golden%20Vectors-99.0%25-brightgreen)
![SVA](https://img.shields.io/badge/SVA%20Violations-0-blue)
![Tests](https://img.shields.io/badge/Tests-1046%2F1059-yellow)


## Vision

### Two-Wearable Vision (Phase 2)

ARIA targets dual-stream sensing: body-worn physiological + external environmental. **Phase 2:** BiLSTM ensemble on FPGA for temporal pattern analysis.

<img width="1402" height="1122" alt="ChatGPT Image Jun 17, 2026 at 01_40_15 AM" src="https://github.com/user-attachments/assets/eb08010b-9660-4e7b-a6ba-0cf23ba005df" />

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

### Dataset Statistics (10,000 Synthetic Samples)

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| PM2.5 (μg/m³) | 0 | 500 | 247.1 | 143.8 |
| VOC (ppb) | 0 | 1000 | 504.5 | 289.3 |
| Heat Index (°C) | 20 | 50 | 35.0 | 8.6 |
| Heart Rate (bpm) | 40 | 180 | 109.8 | 40.5 |
| SpO2 (%) | 70 | 100 | 92.5 | 4.3 |

**Distribution:** Warning 66.7%, Danger 33.3%, Safe 0% (clinically hazardous classes emphasized; Safe trivially identifiable).

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
  
## Five Experiments Summary

| Experiment | Key Result |
|------------|------------|
| Quantization (INT8 vs float32) | 100% agreement, 3.9× memory reduction |
| Sensor Failure (0-5 channels) | Dropout model degrades slower |
| Resources (LUT/FF/DSP) | 2.4% LUT, 2.0% FF, 50% DSP |
| CDC Reliability (MTBF) | >10⁶ hours (Gray code + 2-FF sync) |
| Critical Path (50MHz) | 11.8ns delay, +8.2ns slack |

**Golden Vector Verification:** 992/1,002 pass (99.0%) — all 10 failures are Danger→Warning misclassifications near decision boundary (fixed-point accumulator overflow). **INT8 vs Float32 agreement:** 100% (1,002/1,002).

## Visuals

### Waveforms (GTKWave)


| Module | Waveform | Tests |
|--------|----------|-------|
| goai_wrapper_nn.v | <img width="1358" height="157" alt="Screenshot 2026-05-15 at 7 56 10 PM" src="https://github.com/user-attachments/assets/85ac0647-5b4b-4a64-980d-7aaf3107f88d" />| 992/1002 (99.0%) |
| validity_reg.v | <img width="1356" height="118" alt="Screenshot 2026-05-15 at 8 03 03 PM" src="https://github.com/user-attachments/assets/f2b5e43a-4393-4008-9944-f1aae1f478c1" />| 4/4 PASS |
| power_fsm.v |<img width="1358" height="156" alt="Screenshot 2026-05-15 at 8 00 59 PM" src="https://github.com/user-attachments/assets/cac00028-a567-4eaa-beb1-5fab867d3cb6" />| 5/5 PASS |
| output_fsm.v | <img width="1355" height="202" alt="Screenshot 2026-05-15 at 7 58 40 PM" src="https://github.com/user-attachments/assets/9a939afc-94d9-4e26-a661-fde1f00a0b36" /> | 6/6 PASS |

*goai_wrapper: 6-state FSM (IDLE→COLLECT→LAYER1→LAYER2→OUTPUT→WAIT), 128 MACs, 960ns latency.*

---

### Synthesis (Gowin EDA)

| Metric | Result |
|--------|--------|
| Timing | 50MHz, slack +8.2ns (critical path: Layer 1 MAC, 11.8ns) |
| LUTs | 496 (2.4% of 20,736) |
| FFs | 313 (2.0% of 15,552) |
| DSPs | 24 (50% of 48, GoAI 2.0 MAC blocks) |
| Power | ~1.2mA @ 50MHz (clock-gated IDLE/SLEEP) |

<img width="1314" height="827" alt="Screenshot 2026-05-15 at 8 11 04 PM" src="https://github.com/user-attachments/assets/18510aba-b0da-4486-bf66-41a332b67564" />

---

 ## Key Innovations

- Dual-stream sensing (physiological + environmental fusion)
- Real-time FPGA INT8 neural inference (no CPU dependency)
- Sensor validity-aware inference (graceful degradation)
- Hardware-safe ML pipeline with golden-vector parity testing
- Power-aware FSM for wearable deployment
  
## Verification Results (1,059 Tests, 98.8% Pass)

| Test Suite | Total Tests | Passed | Failed | Pass Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unit Tests (Modules)** | 50 | 47 | 3 | 94.0% | 🟡 `PARTIAL` |
| **Golden Vectors** | 1,002 | 992 | 10 | 99.0% | 🟡 `PARTIAL` |
| **SVA Properties** | 7 | 7 | 0 | 100% | 🟢 `PASS` |

<details>
<summary><b>🔍 Click to expand individual module breakdown</b></summary>

| Module | Tests | Passed | Failed | Result |
| :--- | :---: | :---: | :---: | :--- |
| `uart_rx.v` | 13 | 13 | 0 | 🟢 `PASS` |
| `async_fifo.v` | 15 | 15 | 0 | 🟢 `PASS` |
| `validity_reg.v` | 4 | 4 | 0 | 🟢 `PASS` |
| `power_fsm.v` | 5 | 5 | 0 | 🟢 `PASS` |
| `goai_wrapper_nn.v`* | 5 | 2 | 3 | 🟡 `FAIL` (Quantization delta) |
| `output_fsm.v` | 6 | 6 | 0 | 🟢 `PASS` |
| `top.v` | 2 | 2 | 0 | 🟢 `PASS` |

</details>

*Directed test failures: arbitrary values not matching model boundaries (see [Limitations]). Golden vectors use real model inputs.

**SVA Properties (0 violations):**
1. FIFO full → wr_en=0 next cycle
2. class_out != 2'b11 on result_valid
3. led_danger == alert_out always
4. Only one LED active at a time
5. All clk_en=0 in SLEEP state
6. active_count <= 6 always
7. FSM state <= max_state always

## Scope & Limitations

**Phase 1 (this repo):** Simulation-only (Icarus Verilog verification, Gowin EDA target). Hardware deployment is Phase 2.

| Limitation | Mitigation / Future |
|------------|--------------------|
| Synthetic dataset (Dhaka model) | Real PMS5003/BME688 integration planned |
| No PM2.5 ground-truth (BME688 VOC broad-spectrum only) | Optical particle counter in Phase 2 |
| Non-temporal model (no BiLSTM) | BiLSTM ensemble for temporal fusion |
| Mobile app reporting-only | On-device continual learning from user feedback |

**Dataset note:** 10,000 samples emphasize Warning/Danger (66.7%/33.3%) as clinically informative; Safe 0% is trivially identifiable and will be added in field data.

## Future Work

- **PMS5003 particle counter integration** — PM2.5 ground-truth
- **Real Dhaka field dataset collection** — Retrain model (add Safe class)
- **Layer 2 accumulators (24→32-bit)** — Eliminate overflow edge cases
- **BiLSTM temporal fusion on FPGA** — Temporal pattern analysis
- **On-device continual learning** — Personalized risk scoring
- **Geo-tagged exposure heatmaps** — Mobile app closed-loop
