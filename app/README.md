
---
title: MicroSAM-LoRA
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---


# MicroSAM-LoRA

### Parameter-Efficient Prostate Segmentation on Micro-Ultrasound

An interactive prostate segmentation application powered by
MedSAM fine-tuned using Low-Rank Adaptation (LoRA).

## Overview

MicroSAM-LoRA explores parameter-efficient fine-tuning of
the MedSAM Vision Transformer for prostate segmentation
on micro-ultrasound images.

The application provides an interactive segmentation workflow
where users can upload an image, define a bounding box around
the prostate, and generate a predicted segmentation boundary.

## Model Details

| Component | Configuration |
|---|---|
| Base Model | MedSAM (ViT-B) |
| Fine-tuning | LoRA |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Adapted Layers | Attention QKV and Projection |
| Framework | PyTorch |

## How to Use

1. Upload a micro-ultrasound image.
2. Click two opposite corners around the prostate.
3. Review the bounding-box preview.
4. Click **Segment Prostate**.
5. View the predicted prostate boundary as a red contour.

## Implementation

The application performs the following steps:

- Loads the pretrained MedSAM model.
- Injects trained LoRA adapters into the vision encoder.
- Loads the fine-tuned checkpoint.
- Processes the input image and bounding-box prompt.
- Generates a segmentation mask.
- Applies largest connected-component filtering.
- Extracts the predicted boundary and overlays it in red.

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- OpenCV
- NumPy

## Source Code

[GitHub Repository](https://github.com/ujjawalsingh10/MicroSAM-LoRA)

## Disclaimer

This project is an experimental research demonstration
intended for educational and research purposes.

It is not a clinically validated diagnostic tool and
must not be used for clinical decision-making.
