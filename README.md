# ARIA — Air-Quality Recognition and Inference Accelerator

Real-time air quality classifier running a 2-layer INT8 neural network entirely in hardware on the Tang Nano 20K FPGA (GW2AR-18). No cloud. No external processor.

![Modules](https://img.shields.io/badge/RTL%20Modules-6%2F6-green)
![Golden Vectors](https://img.shields.io/badge/Golden%20Vectors-99.0%25-brightgreen)
![SVA](https://img.shields.io/badge/SVA%20Violations-0-blue)
![Tests](https://img.shields.io/badge/Tests-1046%2F1059-yellow)


## Vision

ARIA is built around a two-wearable concept: one device worn close to the body capturing physiological signals, and one worn externally capturing environmental exposure. The goal is to determine in real time whether pollution, heat, or poor ventilation is contributing to physical symptoms — not just whether air quality is poor in isolation.

| Layer | Node | Sensors |
|---|---|---|
| Exposure | External badge / bag | PM2.5, VOC, Temperature, Humidity, CO2 |
| Response | Wrist / chest wearable | Heart Rate, SpO2, Respiratory Rate, HRV |
| Inference | Tang Nano 20K FPGA | Fused edge classification — Safe / Warning / Danger |
| Feedback | Mobile app | Symptom reports (suffocation, asthma, coughing) |

Fusing both streams lets the system distinguish between high pollution with no physiological effect, high pollution with active health impact, and physiological spikes from exertion rather than exposure. The full system targets a BiLSTM ensemble on the FPGA for temporal pattern analysis. Fixed-point quantization keeps inference within the device's power and memory bounds.

## Current Implementation — Phase 1

This repository implements the FPGA inference core. Phase 1 classifies five fused sensor channels (PM2.5, VOC, Heat Index, HR, SpO2) using a 2-layer INT8 neural network in Verilog RTL on the GW2AR-18. Full two-wearable integration and BiLSTM temporal fusion are planned for Phase 2.


