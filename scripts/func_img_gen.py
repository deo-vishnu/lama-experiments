import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def get_img_1(size, L_ref=128):
    Lx, Ly, _ = size

    x = np.linspace(0, 2*np.pi, Lx, endpoint=False)[:, None]
    y = np.linspace(0, 2*np.pi, Ly, endpoint=False)[None, :]

    s = Lx / L_ref

    img = np.stack([
        np.sin(s * x) * np.cos(3 * s * y),
        np.sin(4 * s * (x + y)),
        np.cos(s * np.sqrt(x**2 + y**2)),
    ], axis=-1).astype(np.float32)

    img -= img.min()
    img /= img.max()

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    H = W = args.size
    size = (H, W, 3)

    img = get_img_1(size)

    print(f"Generated image of shape: {img.shape}")
    print(f"Min: {img.min()}, Max: {img.max()}")

    # Save image if needed
    if args.save:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        save_path = out_dir / f"img_{H}x{W}.npy"
        np.save(save_path, img)

        print(f"Saved to: {save_path}")

if __name__ == "__main__":
    main()