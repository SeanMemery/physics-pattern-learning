from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def write_image_artifact(source: str | Path | Image.Image | np.ndarray, output_path: str | Path, *, max_side: int = 1536) -> dict[str, int]:
    image = _prepare_image(source, max_side=max_side)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return {"width": image.width, "height": image.height}


def write_video_artifact(frames: Iterable[str | Path | Image.Image | np.ndarray], output_path: str | Path, *, fps: int, max_side: int = 1536) -> dict[str, int]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first_source = next(iterator)
    first_image = _ensure_even_video_frame(_prepare_image(first_source, max_side=max_side))
    final_size = first_image.size
    prepared_frames = [np.asarray(first_image)]
    frame_count = 1
    for frame in iterator:
        image = _prepare_image(frame, max_side=max_side)
        if image.size != final_size:
            image = image.resize(final_size, Image.Resampling.LANCZOS)
        prepared_frames.append(np.asarray(_ensure_even_video_frame(image)))
        frame_count += 1
    if frame_count == 1:
        prepared_frames.append(np.asarray(first_image.copy()))
        frame_count += 1
    try:
        writer = imageio.get_writer(destination, fps=fps, codec="libx264", macro_block_size=1)
        try:
            for frame in prepared_frames:
                writer.append_data(frame)
        finally:
            writer.close()
    except Exception:
        _write_video_with_ffmpeg(prepared_frames, destination=destination, fps=fps)
    return {"frame_count": frame_count, "fps": int(fps), "width": int(final_size[0]), "height": int(final_size[1])}


def _write_video_with_ffmpeg(frames: list[np.ndarray], *, destination: Path, fps: int) -> None:
    with tempfile.TemporaryDirectory(prefix="pooltool-rl-frames-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(temp_root / f"frame_{index:05d}.png")
        command = [
            "ffmpeg", "-y", "-framerate", str(int(fps)), "-i", str(temp_root / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffmpeg encode failed")


def _prepare_image(source: str | Path | Image.Image | np.ndarray, *, max_side: int) -> Image.Image:
    image = _load_image(source)
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)


def _load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    pixels = np.asarray(source)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype("uint8")
    if pixels.ndim == 2:
        pixels = np.stack([pixels] * 3, axis=-1)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    return Image.fromarray(pixels, mode="RGB")


def _ensure_even_video_frame(image: Image.Image) -> Image.Image:
    width, height = image.size
    even_width = width + (width % 2)
    even_height = height + (height % 2)
    if (even_width, even_height) == (width, height):
        return image
    padded = Image.new("RGB", (even_width, even_height))
    padded.paste(image, (0, 0))
    return padded
