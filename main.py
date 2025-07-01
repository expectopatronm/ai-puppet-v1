import json5
import time

import numpy as np
from PIL import Image
import torch
from jinja2 import Environment, FileSystemLoader
from google import genai
from google.genai import types

from tha3.util import (
    extract_PIL_image_from_filelike,
    resize_PIL_image,
    extract_pytorch_image_from_PIL_image,
    convert_output_image_from_torch_to_numpy,
)
from tha3.poser.modes.standard_float import create_poser, get_pose_parameters


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
poser = create_poser(DEVICE)  # THA-3 factory
pose_parameters = get_pose_parameters()
POSE_SIZE = poser.get_num_parameters()  # e.g. 45 for “standard_float”

# Frequently accessed parameter indices
iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")
body_y_index = pose_parameters.get_parameter_index("body_y")
body_z_index = pose_parameters.get_parameter_index("body_z")
breathing_index = pose_parameters.get_parameter_index("breathing")


def ease_in_out(t: float) -> float:
    """Smoothstep cubic easing."""
    return t * t * (3 - 2 * t)


def interpolate_poses(keyframes, steps: int = 5):
    """
    Interpolates any numeric entries between consecutive keyframes that have
    `interpolate_to_next=True`, using smoothstep for nicer timing.
    """

    def is_number(x):
        return isinstance(x, (int, float))

    out = []
    for i, cur in enumerate(keyframes):
        nxt = keyframes[i + 1] if i < len(keyframes) - 1 else None
        out.append({k: v for k, v in cur.items() if k != "interpolate_to_next"})
        if nxt and cur.get("interpolate_to_next") and nxt.get("interpolate_to_next"):
            for s in range(1, steps):
                t = ease_in_out(s / steps)
                inter = {}
                for k in cur:
                    if k in ("interpolate_to_next", "mouth_type", "eye_type", "eyebrow_type"):
                        inter[k] = cur[k]
                        continue
                    v1, v2 = cur.get(k, 0.0), nxt.get(k, cur.get(k, 0.0))
                    inter[k] = (1 - t) * v1 + t * v2 if is_number(v1) and is_number(v2) else v1
                out.append(inter)
    return out


def load_image(path: str) -> torch.Tensor:
    """
    Reads RGBA PNG, resizes to 512×512 (model’s native resolution),
    converts to [-1,1] tensor and adds batch dim.
    """
    with open(path, "rb") as f:
        pil = resize_PIL_image(extract_PIL_image_from_filelike(f), size=(512, 512))
    if pil.mode != "RGBA":
        raise ValueError("Image must be RGBA (have an alpha channel).")
    tensor = extract_pytorch_image_from_PIL_image(pil).to(DEVICE)
    return tensor.unsqueeze(0)  # [1,C,H,W]


def create_pose(
    eyebrow_type, eyebrow_left, eyebrow_right,
    eye_type, eye_left, eye_right,
    mouth_type, mouth_left, mouth_right=None,
    iris_small_left=0.0, iris_small_right=0.0,
    iris_rotation_x=0.0, iris_rotation_y=0.0,
    head_x=0.0, head_y=0.0, neck_z=0.0,
    body_y=0.0, body_z=0.0, breathing=0.0,
    exaggerate: float = 1.0,
) -> torch.Tensor:
    """
    Constructs a flat pose vector compatible with the THA-3 animation system.

    This function maps facial and body parameters such as eyebrow movement,
    eye expressions, mouth shapes, iris position, head orientation, and breathing
    into a single pose vector (`torch.Tensor`) suitable for input into the
    THA-3 pose-to-image model.

    Parameters:
        eyebrow_type (str): Type of eyebrow expression (e.g., 'troubled', 'angry').
        eyebrow_left (float): Intensity of the expression on the left eyebrow [0.0–1.0].
        eyebrow_right (float): Intensity of the expression on the right eyebrow [0.0–1.0].

        eye_type (str): Type of eye expression (e.g., 'wink', 'relaxed').
        eye_left (float): Intensity of the expression on the left eye [0.0–1.0].
        eye_right (float): Intensity of the expression on the right eye [0.0–1.0].

        mouth_type (str): Type of mouth shape (e.g., 'aaa', 'smirk', 'lowered_corner').
        mouth_left (float): Intensity for the left side or full mouth if symmetric [0.0–1.0].
        mouth_right (float, optional): Right side intensity for asymmetric shapes.
            Required if `mouth_type` is 'lowered_corner' or 'raised_corner'.

        iris_small_left (float, optional): Shrink amount of the left iris [0.0–1.0].
        iris_small_right (float, optional): Shrink amount of the right iris [0.0–1.0].
        iris_rotation_x (float, optional): Iris rotation around the X-axis [-1.0–1.0].
        iris_rotation_y (float, optional): Iris rotation around the Y-axis [-1.0–1.0].

        head_x (float, optional): Head rotation around the X-axis [-1.0–1.0].
        head_y (float, optional): Head rotation around the Y-axis [-1.0–1.0].
        neck_z (float, optional): Neck twist (Z-axis rotation) [-1.0–1.0].

        body_y (float, optional): Body rotation around the Y-axis [-1.0–1.0].
        body_z (float, optional): Body rotation around the Z-axis [-1.0–1.0].

        breathing (float, optional): Breathing amount [0.0–1.0].

    Returns:
        torch.Tensor: A (1, N) tensor representing the pose vector compatible
        with the poser model.

    Raises:
        ValueError: If `mouth_right` is not provided for asymmetric mouth types
            such as 'lowered_corner' or 'raised_corner'.

    Notes:
        If a given parameter name is not found in the model's pose parameter
        list, the function will emit a warning and skip setting that component.
    """
    pose = torch.zeros(1, POSE_SIZE, dtype=poser.get_dtype(), device=DEVICE)

    def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def apply_exaggeration(value, min_val, max_val):
        amplified = value * (1.0 + exaggerate)
        return clamp(amplified, min_val, max_val)

    def safe_set(parameter_name, value, value_range=(-1.0, 1.0)):
        try:
            idx = pose_parameters.get_parameter_index(parameter_name)
            pose[0, idx] = apply_exaggeration(value, *value_range)
        except RuntimeError:
            print(f"Warning: Parameter '{parameter_name}' not found. Skipping.")

    safe_set(f"eyebrow_{eyebrow_type}_left", eyebrow_left, (0.0, 1.0))
    safe_set(f"eyebrow_{eyebrow_type}_right", eyebrow_right, (0.0, 1.0))
    safe_set(f"eye_{eye_type}_left", eye_left, (0.0, 1.0))
    safe_set(f"eye_{eye_type}_right", eye_right, (0.0, 1.0))

    mname = f"mouth_{mouth_type}"
    if mouth_type in {"lowered_corner", "raised_corner"}:
        if mouth_right is None:
            raise ValueError(f"`mouth_right` is required for mouth_type '{mouth_type}'")
        safe_set(f"{mname}_left", mouth_left, (0.0, 1.0))
        safe_set(f"{mname}_right", mouth_right, (0.0, 1.0))
    else:
        safe_set(mname, mouth_left, (0.0, 1.0))

    safe_set("iris_small_left", iris_small_left, (0.0, 1.0))
    safe_set("iris_small_right", iris_small_right, (0.0, 1.0))
    safe_set("iris_rotation_x", iris_rotation_x, (-1.0, 1.0))
    safe_set("iris_rotation_y", iris_rotation_y, (-1.0, 1.0))
    safe_set("head_x", head_x, (-1.0, 1.0))
    safe_set("head_y", head_y, (-1.0, 1.0))
    safe_set("neck_z", neck_z, (-1.0, 1.0))
    safe_set("body_y", body_y, (-1.0, 1.0))
    safe_set("body_z", body_z, (-1.0, 1.0))
    safe_set("breathing", breathing, (0.0, 1.0))

    return pose


def apply_pose(img: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Runs the poser and returns [C,H,W] RGBA tensor (still linear float)."""
    return poser.pose(img, pose)[0]  # drop batch dim


def animate_sentence(
        image_path: str,
        keyframes: list[dict],
        output_path: str = "animation.gif",
        interp_steps: int = 5,
        fps: int = 15,
        exaggerate: float = 1.0,
):
    """
    Generates an animated GIF by applying pose transformations to a base image.

    Args:
        image_path (str): Path to the base RGBA character image.
        keyframes (list[dict]): List of pose dictionaries to animate through.
        output_path (str): Path to save the resulting GIF animation.
        interp_steps (int): Number of interpolation steps between keyframes.
        fps (int): Frames per second for the final animation.

    Returns:
        None. Saves a GIF to `output_path`.
    """
    keyframes = interpolate_poses(keyframes, interp_steps)
    base_image = load_image(image_path)
    frames = []

    for kf in keyframes:
        pose = create_pose(
            eyebrow_type=kf.get("eyebrow_type", "serious"),
            eyebrow_left=kf.get("eyebrow_left", 0.0),
            eyebrow_right=kf.get("eyebrow_right", 0.0),
            eye_type=kf.get("eye_type", "relaxed"),
            eye_left=kf.get("eye_left", 0.5),
            eye_right=kf.get("eye_right", 0.5),
            mouth_type=kf.get("mouth_type", "aaa"),
            mouth_left=kf.get("mouth_left", 0.5),
            mouth_right=kf.get("mouth_right", kf.get("mouth_left", 0.5)),
            iris_small_left=kf.get("iris_small_left", 0.0),
            iris_small_right=kf.get("iris_small_right", 0.0),
            iris_rotation_x=kf.get("iris_rotation_x", 0.0),
            iris_rotation_y=kf.get("iris_rotation_y", 0.0),
            head_x=kf.get("head_x", 0.0),
            head_y=kf.get("head_y", 0.0),
            neck_z=kf.get("neck_z", 0.0),
            body_y=kf.get("body_y", 0.0),
            body_z=kf.get("body_z", 0.0),
            breathing=kf.get("breathing", 0.0),
            exaggerate=exaggerate,
        )
        rgba = apply_pose(base_image, pose)

        np_img = np.uint8(
            np.rint(convert_output_image_from_torch_to_numpy(rgba.detach().cpu()) * 255.0)
        )
        pil_rgba = Image.fromarray(np_img, mode="RGBA")

        rgb = Image.new("RGB", pil_rgba.size, (255, 255, 255))
        rgb.paste(pil_rgba, mask=pil_rgba.split()[3])
        frames.append(rgb)

    if frames:
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            format="GIF",
        )
        print(f"✅ GIF saved to: {output_path}")
    else:
        print("❌ No frames generated, GIF not created.")


if __name__ == "__main__":
    import os
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("prompt_template.txt")
    prompt = template.render(sentence="Navigation to the nearest LIDL has been initiated!")

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    t0 = time.time()
    llm_resp = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            response_mime_type="application/json",
        ),
    )
    print(f"LLM call took {time.time() - t0:.1f} s")

    key_poses = json5.loads(llm_resp.text)

    t0 = time.time()
    animate_sentence(
        image_path="data/images/crypko_05.png",
        keyframes=key_poses,
        exaggerate=0.75,
        interp_steps=5,
        output_path="animation_tha3.gif",
    )
    print(f"Animation generation took {time.time() - t0:.1f} s")
