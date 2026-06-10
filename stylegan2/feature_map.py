# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
"""Extract and visualize StyleGAN synthesis feature maps."""

import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


def num_range(s: str) -> List[int]:
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(",")]


def normalize_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return (x * 255).to(torch.uint8).numpy()


def save_feature_grid(feat: torch.Tensor, save_path: str, max_channels: int = 64):
    # feat: [C, H, W]
    C, H, W = feat.shape
    n = min(C, max_channels)
    grid_size = int(np.ceil(np.sqrt(n)))

    canvas = np.zeros((grid_size * H, grid_size * W), dtype=np.uint8)

    for i in range(n):
        r = i // grid_size
        c = i % grid_size
        ch_img = normalize_to_uint8(feat[i])
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = ch_img

    PIL.Image.fromarray(canvas, mode="L").save(save_path)


def save_mean_activation(feat: torch.Tensor, save_path: str):
    # feat: [C, H, W]
    mean_map = feat.abs().mean(dim=0)
    img = normalize_to_uint8(mean_map)
    PIL.Image.fromarray(img, mode="L").save(save_path)


def save_generated_image(img: torch.Tensor, save_path: str):
    # img: [1, 3, H, W], range usually [-1, 1]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128)
    img = img.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(img, mode="RGB").save(save_path)


@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle filename")
@click.option("--rows", "row_seeds", type=num_range, required=True, help="Random seeds, e.g. 1,2,3 or 1-10")
@click.option("--trunc", "truncation_psi", type=float, default=1, show_default=True)
@click.option("--noise-mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--outdir", type=str, required=True)
@click.option(
    "--layers",
    type=str,
    default="synthesis.b16.conv1,synthesis.b32.conv1,synthesis.b64.conv1,synthesis.b128.conv1,synthesis.b256.conv1",
    show_default=True,
    help="Comma-separated layer names to hook",
)
@click.option("--max-channels", type=int, default=64, show_default=True)
def feature_map(
    network_pkl: str,
    row_seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    layers: str,
    max_channels: int,
):
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device("cuda")

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    os.makedirs(outdir, exist_ok=True)

    target_layers = [x.strip() for x in layers.split(",") if x.strip()]
    features = {}

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                features[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                features[name] = output[0].detach()
        return hook

    hooks = []
    existing_layer_names = {name for name, _ in G.named_modules()}

    print("Registering hooks...")
    for layer_name in target_layers:
        if layer_name not in existing_layer_names:
            print(f"WARNING: layer not found: {layer_name}")
            continue

        module = dict(G.named_modules())[layer_name]
        hooks.append(module.register_forward_hook(make_hook(layer_name)))
        print(f"Hooked: {layer_name}")

    if len(hooks) == 0:
        raise RuntimeError("No valid layers were hooked. Check --layers names.")

    print("Available synthesis layers:")
    for name, module in G.named_modules():
        if type(module).__name__ == "SynthesisLayer":
            print(" ", name)

    print("Extracting feature maps...")

    for seed in row_seeds:
        features.clear()

        z = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z).to(device)

        with torch.no_grad():
            w = G.mapping(z, None)
            w_avg = G.mapping.w_avg
            w = w_avg + (w - w_avg) * truncation_psi

            img = G.synthesis(w, noise_mode=noise_mode)

        seed_dir = os.path.join(outdir, f"seed{seed:04d}_features")
        os.makedirs(seed_dir, exist_ok=True)

        save_generated_image(img, os.path.join(seed_dir, f"seed{seed:04d}.png"))

        for layer_name, feat_batch in features.items():
            feat = feat_batch[0]  # [C, H, W]
            safe_name = layer_name.replace(".", "_")

            torch.save(feat.cpu(), os.path.join(seed_dir, f"{safe_name}.pt"))
            np.save(os.path.join(seed_dir, f"{safe_name}.npy"), feat.cpu().numpy())

            save_feature_grid(
                feat,
                os.path.join(seed_dir, f"{safe_name}_channels.png"),
                max_channels=max_channels,
            )

            save_mean_activation(
                feat,
                os.path.join(seed_dir, f"{safe_name}_mean_abs.png"),
            )

        print(f"Saved seed {seed} feature maps to {seed_dir}")

    for h in hooks:
        h.remove()

    print("Done.")


if __name__ == "__main__":
    feature_map()