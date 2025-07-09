# THA3-Driven Expressive Facial Animation  
**Speech-to-animation pipeline with emotion, visemes, and upper-body dynamics — powered by THA3 + LLMs**

![Demo Animation](data/outputs/animation_tha3.gif)


## Goal

Create compelling and emotionally expressive 2D character animations **from spoken text** using the THA3 model. Each animation mimics the natural rhythm of speech by blending:

- **Lip-sync (visemes + coarticulation)**
- **Emotion (facial expression, eye shape, mouth tweaks)**
- **Subtle body motion (posture, head tilt, breathing)**
- **Timing finesse (blinks, focus shifts, emphasis)**

This system produces only **10–20 expressive keyframes** per sentence, which are then **smoothly interpolated**, achieving high emotional fidelity with low frame count.

---

## What Makes It Different?

Unlike traditional talking-heads animation that rely purely on phoneme-to-mouth mapping, this system:
- Embeds **emotion and tone** via brows, eyes, head/body  
- Adds **gaze dynamics, blinks, breathing**  
- Uses a **viseme+emotion composition model**  
- Follows **coarticulation rules** and **speech rhythm**  
- Generates animation plans using a **language model prompt** grounded in animation principles

---

## Features

- 🎙 **LLM-driven animation planning** (Gemini 2.5)
- 👄 **Viseme-aligned lip sync** using standard phoneme groups
- 🤨 **Emotional mouth/brow overlays** (smirk, smile, raised_corner, etc.)
- 👀 **Eye blinking, iris tracking, gaze aversion**
- 🫁 **Breathing modulation** linked to emotion
- 💃 **Head tilt, posture shift, lean-in emphasis**
- 🧘 **Relaxed idle keyframes** at start and end
- 🔁 **Interpolated motion** using smoothstep easing
- 🎞 **Output as GIF**, built with THA3-compatible pose vectors

---

## Prompt Design & Research Rationale

### 1. **Phoneme → Viseme Mapping**
Based on established speech animation literature (e.g., Parke & Waters, 1996), phonemes are grouped into visual equivalents:

| Phoneme Group     | Viseme Type |
|-------------------|-------------|
| AA, AH, AO, AE    | `aaa`       |
| IY, IH            | `iii`       |
| UW, UH            | `uuu`       |
| EH, EY            | `eee`       |
| OW, OY            | `ooo`       |
| M, B, P, N, D, T  | `delta`     |
| Silences / Pauses | `delta`     |

These are mapped to the `mouth_type` parameter used by THA3.

### 2. **Emotion Mapping**
Facial expressions are derived from **punctuation**, **tone**, and **semantic cues**:

- `!` → **excitement**, use `happy_wink`, `raised_corner`, fast breathing
- `.` → **neutral or relaxed**, soft blink, lower breathing
- `?` → **questioning**, with raised eyebrows, head tilt (neck_z), and surprised eyes

This mimics animation acting principles such as "anticipation", "accent", and "reaction".

### 3. **Coarticulation Rules**
To avoid robotic frame-to-frame switching:

- Consecutive visemes **blend smoothly**
- Emotional mouth shapes (e.g., smile) are **layered** over visemes
- Mouth_left and mouth_right allow **asymmetric shaping** during transition

### 4. **Head, Neck & Body Language**
Inspired by real-time avatar animation (e.g., VRM, VUP), movement cues are used to simulate attention and expression:

| Component | Purpose |
|----------|---------|
| `head_y` | nodding, emphasis |
| `neck_z` | head tilt for questions or sarcasm |
| `body_y` | body sway (left/right weight shift) |
| `body_z` | lean-in/lean-back posture dynamics |

Breathing (`breathing`) is modulated based on **energy** or **emotion**.

### 5. **Eye Dynamics**
Blinks and eye focus are critical for human realism.

- Expressive blinks every **2–4 seconds**
- Occasional **double blink** for emotion beats
- Iris rotates subtly ±0.1–0.3 to simulate gaze
- `eye_type` toggles between `relaxed`, `surprised`, `wink`, etc.

### 6. **Keyframe Strategy**
Only **10–20 keyframes per sentence**, each for:

- Viseme switch
- Blink
- Emotional shift
- Gaze change
- Posture reweight

All are marked with `"interpolate_to_next": true`, except final.

---

## Core Components

### `create_pose(...)`
Encodes all facial and body parameters (eyebrows, eyes, mouth, iris size/rotation, head tilt, neck twist, body lean, breathing) into a `torch.Tensor` compatible with the THA3 model.

This function allows:
- Simple high-level API use (e.g., `mouth_type="aaa"`, `eyebrow_type="raised"`)
- Asymmetric expressions (`mouth_left`, `mouth_right`)
- Pose exaggeration (via a float multiplier)
- Validation and clamping within THA3’s pose parameter ranges

---

### `interpolate_poses(...)`
Smoothly interpolates between each pair of keyframes using cubic easing (`smoothstep`). This:
- Ensures natural coarticulation between visemes
- Preserves emotion over time
- Prevents robotic “jump cuts” in expression

Each intermediate frame blends:
- All numeric values (like `head_y`, `breathing`, etc.)
- While keeping expression types (`eye_type`, `mouth_type`) consistent per step

---

### `animate_sentence(...)`
Main rendering loop that:
- Loads a base image (RGBA 512×512)
- Applies each interpolated pose using `poser.pose(...)`
- Converts the output frame to PIL and collects all frames
- Saves animation as a `.gif` (typically 2–5 seconds long)

---

### `prompt_template.txt`
This is the master **animation scripting prompt**, passed to an LLM (Google Gemini). It includes:
- Full function interface docstring (for `create_pose`)
- Viseme mapping table
- Detailed behavioral rules (emotion, eye direction, head/body motion, breathing, blinks)
- Output format structure
- Emphasis on “expressive beats” and natural speech timing

---

### `animate.py`
Script that brings everything together:
- Renders the prompt using Jinja2
- Sends it to the LLM with Gemini API
- Parses the returned keyframes
- Interpolates and renders the final animation

---

## License & Credits

**License:** MIT

**Credits:**
- Core animation model and poser logic by [THA3 (TorchHub Animation 3)](https://github.com/hyperpose/THA3)
- Animation logic and expressive pose authoring developed on top of THA3’s APIs
- Prompt-based pose generation powered by Google Gemini LLM
- Character portrait from [Crypko.ai](https://crypko.ai) (non-commercial use only)

---

## Acknowledgments

- [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo)): Torch-based 2D poser and avatar animation system
- [Google Gemini](https://ai.google.dev): High-speed LLM used for pose planning
- Foundational works in animation research:
  - Cassell et al., “Animated Conversation” (1994)
  - Parke & Waters, “Computer Facial Animation” (1996)
  - Beskow, “Rule-based visual speech synthesis” (2003)
