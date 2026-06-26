from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from .rollout import DEFAULT_CALVIN_ROOT, load_eval_components, make_env


def _image_from_obs(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CALVIN scene-D rollout observation snapshot.")
    parser.add_argument("--calvin-root", default=str(DEFAULT_CALVIN_ROOT))
    parser.add_argument("--scene", default="calvin_scene_D_eval")
    parser.add_argument("--sequence-index", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--use-egl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    get_sequences, _, val_annotations = load_eval_components(args.calvin_root)
    sequences = get_sequences(1000)
    if args.sequence_index < 0 or args.sequence_index >= len(sequences):
        raise ValueError(f"sequence-index must be in [0, {len(sequences) - 1}]")

    initial_state, eval_sequence = sequences[args.sequence_index]
    first_task = eval_sequence[0]
    annotation = val_annotations[first_task][0]

    env = make_env(
        calvin_root=args.calvin_root,
        scene=args.scene,
        show_gui=args.show_gui,
        use_egl=args.use_egl,
    )
    try:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        static = _image_from_obs(obs["rgb_obs"]["rgb_static"])
        gripper = _image_from_obs(obs["rgb_obs"]["rgb_gripper"])
    finally:
        env.close()
        env.ownsPhysicsClient = False
        env.cid = -1

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), dpi=200, constrained_layout=True)
    for axis, image, title in zip(axes, [static, gripper], ["Static camera", "Gripper camera"], strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#555555")

    readable_task = first_task.replace("_", " ")
    fig.suptitle(f"CALVIN scene D rollout observation: {readable_task}", fontsize=11)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"{output}\nannotation: {annotation}")


if __name__ == "__main__":
    main()
