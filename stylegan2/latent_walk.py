# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
"""Generate latent walk images and average-z image using pretrained network pickle."""

import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    """Accept comma-separated numbers 'a,b,c' or range 'a-c'."""
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(",")]


def lerp(a, b, t):
    return a + (b - a) * t


def slerp(a, b, t, eps=1e-8):
    a_norm = a / (a.norm() + eps)
    b_norm = b / (b.norm() + eps)

    dot = torch.clamp(torch.sum(a_norm * b_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    if torch.abs(omega) < eps:
        return lerp(a, b, t)

    so = torch.sin(omega)
    return torch.sin((1.0 - t) * omega) / so * a + torch.sin(t * omega) / so * b


def tensor_to_pil(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")


def save_grid(images, out_path, grid_w):
    W, H = images[0].size
    grid_h = int(np.ceil(len(images) / grid_w))

    canvas = PIL.Image.new("RGB", (W * grid_w, H * grid_h), "black")
    for idx, img in enumerate(images):
        x = (idx % grid_w) * W
        y = (idx // grid_w) * H
        canvas.paste(img, (x, y))

    canvas.save(out_path)


#----------------------------------------------------------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle filename")
@click.option("--seeds", type=num_range, required=True, help="Seed sequence, e.g. 1,2,3 or 0-10")
@click.option("--space", type=click.Choice(["z", "w"]), default="w", show_default=True)
@click.option("--interp", type=click.Choice(["linear", "spherical"]), default="linear", show_default=True)
@click.option("--num-steps", type=int, default=16, show_default=True)
@click.option("--trunc", "truncation_psi", type=float, default=1, show_default=True)
@click.option("--noise-mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--outdir", type=str, required=True)
@click.option("--save-frames/--no-save-frames", default=True, show_default=True)
@click.option("--save-grid/--no-save-grid", default=True, show_default=True)
@click.option("--save-average-z/--no-save-average-z", default=True, show_default=True)
def generate_latent_walk(
    network_pkl: str,
    seeds: List[int],
    space: str,
    interp: str,
    num_steps: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    save_frames: bool,
    save_grid: bool,
    save_average_z: bool,
):
    if len(seeds) < 2:
        raise click.ClickException("--seeds must contain at least two seeds.")

    if num_steps < 2:
        raise click.ClickException("--num-steps must be at least 2.")

    if space == "w" and interp == "spherical":
        print("warn: spherical interpolation is only used for z-space. Switching to linear.")
        interp = "linear"

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device("cuda")

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    frames_root = os.path.join(outdir, "frames")
    walk_image_root = os.path.join(outdir, "latent_walk_image")
    average_root = os.path.join(outdir, "average_z_image")

    if save_frames:
        os.makedirs(frames_root, exist_ok=True)
    if save_grid:
        os.makedirs(walk_image_root, exist_ok=True)
    if save_average_z:
        os.makedirs(average_root, exist_ok=True)

    print("Generating z and w vectors...")

    z_dict = {}
    w_dict = {}

    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z_dict[seed] = z

        w = G.mapping(z, None)
        w_avg = G.mapping.w_avg
        w = w_avg + (w - w_avg) * truncation_psi
        w_dict[seed] = w

    # ------------------------------------------------------------
    # Average-z image from all given seeds.
    # ------------------------------------------------------------

    if save_average_z:
        print("Generating image from average z of all given seeds...")

        avg_z = torch.stack([z_dict[seed][0] for seed in seeds], dim=0).mean(dim=0, keepdim=True)

        avg_img = G(
            avg_z,
            None,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
        )

        avg_pil = tensor_to_pil(avg_img)
        seed_name = "_".join([str(seed) for seed in seeds])
        avg_pil.save(os.path.join(average_root, f"average_z_{seed_name}.png"))

    # ------------------------------------------------------------
    # Latent walks between consecutive seeds.
    # ------------------------------------------------------------

    print("Generating latent walks...")

    for pair_idx in range(len(seeds) - 1):
        seed_a = seeds[pair_idx]
        seed_b = seeds[pair_idx + 1]

        print(f"Walking from seed {seed_a} to seed {seed_b}...")

        pair_name = f"{seed_a:04d}_to_{seed_b:04d}"
        pair_frame_dir = os.path.join(frames_root, pair_name)

        if save_frames:
            os.makedirs(pair_frame_dir, exist_ok=True)

        images = []

        for step_idx in range(num_steps):
            t_float = step_idx / (num_steps - 1)
            t = torch.tensor(t_float, device=device)

            if space == "z":
                za = z_dict[seed_a][0]
                zb = z_dict[seed_b][0]

                if interp == "spherical":
                    z = slerp(za, zb, t).unsqueeze(0)
                else:
                    z = lerp(za, zb, t).unsqueeze(0)

                img = G(
                    z,
                    None,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode,
                )

            else:
                wa = w_dict[seed_a]
                wb = w_dict[seed_b]
                w = lerp(wa, wb, t)

                img = G.synthesis(
                    w,
                    noise_mode=noise_mode,
                )

            pil_img = tensor_to_pil(img)
            images.append(pil_img)

            if save_frames:
                pil_img.save(os.path.join(pair_frame_dir, f"frame{step_idx:03d}.png"))

        if save_grid:
            grid_path = os.path.join(walk_image_root, f"{pair_name}.png")
            save_grid_fn = globals()["save_grid"]
            save_grid_fn(images, grid_path, grid_w=num_steps)

    print("Done.")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_latent_walk()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------