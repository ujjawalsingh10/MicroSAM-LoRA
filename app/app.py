import copy
import traceback
import spaces
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
MODEL_ID = "wanglab/medsam-vit-base"
APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(
    os.environ.get(
        "LORA_CKPT_PATH",
        str(APP_DIR / "medsam_lora_r8_best.pth")
    )
)
EXAMPLE_DIR = APP_DIR / "examples"
LORA_RANK = 8
LORA_ALPHA = 16

print(f"ZeroGPU environment: {os.environ.get('SPACES_ZERO_GPU')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")


# ---------------------------------------------------------
# 2. LoRA layer
# ---------------------------------------------------------
class LoRALinear(nn.Module):
    """Frozen Linear layer plus a trainable low-rank update."""

    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        device = base_linear.weight.device

        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)


def inject_lora(model, r=8, alpha=16):
    for layer in model.vision_encoder.layers:
        layer.attn.qkv = LoRALinear(layer.attn.qkv, r=r, alpha=alpha)
        layer.attn.proj = LoRALinear(layer.attn.proj, r=r, alpha=alpha)
    return model


def prepare_medsam_lora(model, r=8, alpha=16):
    for p in model.parameters():
        p.requires_grad = False
    model = inject_lora(model, r=r, alpha=alpha)
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    return model


# ---------------------------------------------------------
# 3. Load model + checkpoint (CPU copy first, GPU copy second)
# ---------------------------------------------------------
def load_model():
    """Returns the fine-tuned model on CPU, plus the processor."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    processor = SamProcessor.from_pretrained(MODEL_ID)
    model = SamModel.from_pretrained(MODEL_ID)
    model = prepare_medsam_lora(model, r=LORA_RANK, alpha=LORA_ALPHA)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        state_dict[key] = value

    incompatible = model.load_state_dict(state_dict, strict=False)

    # Fail loudly instead of silently running with untrained adapters.
    missing_lora = [k for k in incompatible.missing_keys if "lora_A" in k or "lora_B" in k]
    if missing_lora:
        raise RuntimeError(f"LoRA weights not loaded: {missing_lora[:10]}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:10]}")

    model.eval()
    print("MedSAM-LoRA loaded successfully (CPU).")
    return model, processor


cpu_model, processor = load_model()

# Separate GPU copy, used first. The CPU copy is the fallback
# when the ZeroGPU quota is exhausted.
gpu_model = None
if torch.cuda.is_available():
    gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
    print("GPU copy ready.")


# ---------------------------------------------------------
# 4. Bounding-box helpers
# ---------------------------------------------------------
START_MESSAGE = "**Step 1:** upload an image (or pick a sample below), then click two opposite corners around the prostate."


def draw_box_preview(image, points):
    if image is None:
        return None
    preview = image.copy()
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    elif len(points) == 1:
        x, y = points[0]
        cv2.circle(preview, (x, y), 5, (0, 255, 0), -1)
    return preview


def register_click(image, points, event: gr.SelectData):
    if image is None:
        return points, None, "Upload an image first."

    if len(points) >= 2:
        points = []

    x, y = event.index
    height, width = image.shape[:2]
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))

    points = points + [(x, y)]
    preview = draw_box_preview(image, points)

    if len(points) == 1:
        message = "**Step 2:** first corner set. Click the opposite corner."
    else:
        message = "**Step 3:** box ready (see right panel). Click **Segment Prostate**."
    return points, preview, message


def reset_box(image):
    return [], image, "Box reset. Click two new corners."


def on_new_image(_image):
    # Fires when the input image changes (upload, sample, or clear).
    return [], None, START_MESSAGE


def clear_all():
    return None, [], None, START_MESSAGE


# ---------------------------------------------------------
# 5. Post-processing
# ---------------------------------------------------------
def keep_largest_component(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if count <= 1:
        return binary_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def draw_red_contour(image, binary_mask):
    output = image.copy()
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
    return output


# ---------------------------------------------------------
# 6. Inference (GPU first, CPU fallback)
# ---------------------------------------------------------
def run_inference(net, device, pil_image, box):
    """Runs the model on the given device and returns an (H, W) uint8 mask."""
    inputs = processor(
        images=pil_image,
        input_boxes=[[box]],
        return_tensors="pt"
    )
    original_sizes = inputs["original_sizes"]
    reshaped_input_sizes = inputs["reshaped_input_sizes"]

    outputs = net(
        pixel_values=inputs["pixel_values"].to(device),
        input_boxes=inputs["input_boxes"].to(device),
        multimask_output=False,
    )

    predicted_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        reshaped_input_sizes
    )
    mask_logits = predicted_masks[0][0, 0]
    return (mask_logits.numpy() > 0).astype(np.uint8)


@spaces.GPU(duration=30)
def segment_on_gpu(pil_image, box):
    with torch.inference_mode():
        return run_inference(gpu_model, torch.device("cuda"), pil_image, box)


def segment_prostate(image, points):
    if image is None:
        raise gr.Error("Please upload a micro-ultrasound image.")
    if points is None or len(points) != 2:
        raise gr.Error("Please select two opposite corners to create a bounding box.")

    image_rgb = image.astype(np.uint8)
    pil_image = Image.fromarray(image_rgb)

    (x1, y1), (x2, y2) = points
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    mask = None
    used = "GPU"

    if gpu_model is not None:
        try:
            mask = segment_on_gpu(pil_image, box)
        except Exception:
            # Most often: ZeroGPU quota reached. Log it and fall back.
            traceback.print_exc()
            gr.Warning("GPU unavailable (quota may be used up). Running on CPU instead, which is slower.")

    if mask is None:
        used = "CPU"
        with torch.inference_mode():
            mask = run_inference(cpu_model, torch.device("cpu"), pil_image, box)

    mask = keep_largest_component(mask)
    result = draw_red_contour(image_rgb, mask)
    return result, f"**Done** (ran on {used}). Click new corners to try another box, or **Clear** to start over."


# ---------------------------------------------------------
# 7. Gradio interface
# ---------------------------------------------------------
IMG_HEIGHT = 420

with gr.Blocks() as demo:
    gr.Markdown(
        "# 🩻 MicroSAM-LoRA\n"
        "### Interactive Prostate Segmentation on Micro-Ultrasound"
    )
    status = gr.Markdown(START_MESSAGE)

    box_state = gr.State([])

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="1. Input image (click two opposite corners)",
                type="numpy",
                height=IMG_HEIGHT,
            )
            with gr.Row():
                reset_button = gr.Button("Reset Box")
                clear_button = gr.Button("Clear")
            segment_button = gr.Button("Segment Prostate", variant="primary")
        with gr.Column():
            output_image = gr.Image(
                label="2. Box preview / segmentation result",
                type="numpy",
                interactive=False,
                height=IMG_HEIGHT,
            )

    # Sample images: put up to 3 files in an "examples" folder next to app.py.
    example_files = []
    if EXAMPLE_DIR.is_dir():
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            example_files.extend(sorted(EXAMPLE_DIR.glob(pattern)))
    example_files = [[str(p)] for p in example_files[:3]]

    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=[image_input],
            label="Sample images (click one to load it)",
        )

    gr.Markdown(
        """
        ---
        **Research demonstration only.** This tool is intended for
        research and educational use. It is not a clinically validated
        diagnostic system and should not be used for medical decisions.
        """
    )

    # New upload / sample / clear: reset the box and the right panel.
    image_input.change(
        fn=on_new_image,
        inputs=[image_input],
        outputs=[box_state, output_image, status],
    )
    image_input.select(
        fn=register_click,
        inputs=[image_input, box_state],
        outputs=[box_state, output_image, status],
    )
    reset_button.click(
        fn=reset_box,
        inputs=[image_input],
        outputs=[box_state, output_image, status],
    )
    clear_button.click(
        fn=clear_all,
        outputs=[image_input, box_state, output_image, status],
    )
    segment_button.click(
        fn=segment_prostate,
        inputs=[image_input, box_state],
        outputs=[output_image, status],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)