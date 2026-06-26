from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm import tqdm

from .calvin_dataset import _load_image_array
from .language import encode_language, language_text, load_task_embeddings
from .utils import configure_torch_runtime, json_dump, resolve_device, set_seed


DEFAULT_CALVIN_ROOT = Path("/home/zzh/Titan/v12/repos/reference/calvin")
EP_LEN = 360


class ACTCalvinModel:
    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        image_size: int,
        language_embedding_backend: str = "calvin_sbert",
        task_embeddings_path: str | Path | None = None,
        language_source: str = "annotation",
        temporal_ensemble_coeff: float | None = None,
    ) -> None:
        from lerobot.policies.act import ACTPolicy
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        self.device = device
        self.image_size = image_size
        self.language_embedding_backend = language_embedding_backend
        self.language_source = language_source
        self.task_embeddings_path = Path(task_embeddings_path).expanduser().resolve() if task_embeddings_path else None
        self.task_embeddings: dict[str, np.ndarray] | None = None
        self.policy = ACTPolicy.from_pretrained(Path(checkpoint), local_files_only=True, strict=False)
        self.policy.config.device = str(device)
        if temporal_ensemble_coeff is not None:
            self.policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                chunk_size=self.policy.config.chunk_size,
            )
        self.policy.to(device)
        self.policy.eval()
        self.expected_state_dim = int(self.policy.config.robot_state_feature.shape[0])
        self.robot_state_dim = 15
        self.language_embedding_dim = max(0, self.expected_state_dim - self.robot_state_dim)
        if self.expected_state_dim < self.robot_state_dim:
            raise ValueError(f"Checkpoint expects state dim {self.expected_state_dim}, below CALVIN robot dim 15")
        if self.language_embedding_dim > 0 and self.language_embedding_backend == "calvin_sbert":
            if self.task_embeddings_path is None:
                raise ValueError("CALVIN SBERT language-conditioned ACT rollout requires --task-embeddings-path")
            self.task_embeddings = load_task_embeddings(self.task_embeddings_path, dim=self.language_embedding_dim)

    def reset(self) -> None:
        self.policy.reset()

    def format_goal(self, *, task: str, annotation: str) -> dict[str, str]:
        return {
            "task": task,
            "annotation": annotation,
            "text": language_text(task=task, annotation=annotation, source=self.language_source),
        }

    def _state_with_language(self, robot_obs: np.ndarray, goal: str | dict[str, str] | None) -> np.ndarray:
        state = robot_obs.astype(np.float32, copy=False)
        if self.language_embedding_dim == 0:
            return state
        if goal is None:
            raise ValueError("Language-conditioned ACT checkpoint requires a language goal string")
        if self.language_embedding_backend == "calvin_sbert":
            if self.task_embeddings is None:
                raise RuntimeError("CALVIN task embeddings were not loaded")
            if not isinstance(goal, dict):
                raise ValueError("CALVIN SBERT rollout requires a structured goal with a task key")
            task = goal["task"]
            if task not in self.task_embeddings:
                raise KeyError(f"Task {task!r} missing from {self.task_embeddings_path}")
            lang = self.task_embeddings[task]
        else:
            text = goal["text"] if isinstance(goal, dict) else str(goal)
            lang = encode_language(text, self.language_embedding_dim)
        return np.concatenate([state, lang], axis=0).astype(np.float32, copy=False)

    def step(self, obs: dict[str, Any], goal: str | None = None) -> np.ndarray:
        batch = {
            "observation.state": torch.from_numpy(self._state_with_language(obs["robot_obs"], goal))
            .unsqueeze(0)
            .to(self.device),
            "observation.images.static": _load_image_array(obs["rgb_obs"]["rgb_static"], self.image_size)
            .unsqueeze(0)
            .to(self.device),
            "observation.images.gripper": _load_image_array(obs["rgb_obs"]["rgb_gripper"], self.image_size)
            .unsqueeze(0)
            .to(self.device),
        }
        action = self.policy.select_action(batch).squeeze(0).detach().cpu().numpy().astype(np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
        return action


def _calvin_paths(calvin_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(calvin_root).expanduser().resolve()
    env_src = root / "calvin_env"
    models_src = root / "calvin_models"
    if not env_src.is_dir() or not models_src.is_dir():
        raise FileNotFoundError(f"Expected calvin_env and calvin_models under {root}")
    return root, env_src, models_src


def make_env(
    *,
    calvin_root: str | Path,
    scene: str,
    show_gui: bool = False,
    use_egl: bool = False,
):
    import calvin_env

    _, env_src, _ = _calvin_paths(calvin_root)
    conf_dir = Path(calvin_env.__file__).parents[1] / "conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(
            config_name="config_data_collection",
            overrides=[
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
                "cameras=static_and_gripper",
                f"scene={scene}",
                "robot=panda_longer_finger",
                "use_vr=false",
                "record=false",
                f"data_path={env_src / 'data'}",
                f"env.use_egl={'true' if use_egl else 'false'}",
                f"env.show_gui={'true' if show_gui else 'false'}",
            ],
        )
    OmegaConf.resolve(cfg)
    return hydra.utils.instantiate(cfg.env, show_gui=show_gui, use_vr=False, use_scene_info=True)


def load_eval_components(calvin_root: str | Path):
    _, _, models_src = _calvin_paths(calvin_root)
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    conf_dir = models_src / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return get_sequences, task_oracle, val_annotations


def count_success(results: list[int]) -> list[float]:
    if not results:
        return [float("nan")] * 5
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        step_success.append(n_success / len(results))
    return step_success


def get_task_stats(results: list[int], sequences: list[tuple[dict, list[str]]]) -> dict[str, dict[str, int | float]]:
    cnt_success: Counter[str] = Counter()
    cnt_fail: Counter[str] = Counter()
    for result, (_, sequence) in zip(results, sequences):
        for successful_task in sequence[:result]:
            cnt_success[successful_task] += 1
        if result < len(sequence):
            cnt_fail[sequence[result]] += 1
    total = cnt_success + cnt_fail
    return {
        task: {
            "success": int(cnt_success[task]),
            "total": int(total[task]),
            "success_rate": float(cnt_success[task] / total[task]) if total[task] else float("nan"),
        }
        for task in sorted(total)
    }


def evaluate_sequence(
    *,
    env,
    model: ACTCalvinModel,
    task_oracle,
    initial_state: dict,
    eval_sequence: list[str],
    val_annotations,
    ep_len: int,
) -> int:
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask in eval_sequence:
        lang_annotation = val_annotations[subtask][0]
        goal = (
            model.format_goal(task=subtask, annotation=lang_annotation)
            if hasattr(model, "format_goal")
            else lang_annotation
        )
        model.reset()
        start_info = env.get_info()
        success = False
        for _ in range(ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break
        if not success:
            return success_counter
        success_counter += 1
    return success_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop CALVIN D rollouts for a LeRobot ACT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--calvin-root", default=str(DEFAULT_CALVIN_ROOT))
    parser.add_argument("--scene", default="calvin_scene_D_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--language-source",
        default="annotation",
        choices=["annotation", "task", "task_annotation"],
        help="How to format rollout goals for language-conditioned ACT checkpoints.",
    )
    parser.add_argument("--language-embedding-backend", default="calvin_sbert", choices=["hash", "calvin_sbert"])
    parser.add_argument("--task-embeddings-path", default=None)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Enable ACT temporal ensembling at inference. Original ACT commonly uses 0.01.",
    )
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--start-sequence", type=int, default=0)
    parser.add_argument("--ep-len", type=int, default=EP_LEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--use-egl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_runtime()
    set_seed(args.seed)
    device = resolve_device(args.device)

    get_sequences, task_oracle, val_annotations = load_eval_components(args.calvin_root)
    all_sequences = get_sequences(1000)
    selected = all_sequences[args.start_sequence : args.start_sequence + args.num_sequences]
    if not selected:
        raise ValueError("No CALVIN evaluation sequences selected")

    model = ACTCalvinModel(
        args.checkpoint,
        device=device,
        image_size=args.image_size,
        language_embedding_backend=args.language_embedding_backend,
        task_embeddings_path=args.task_embeddings_path,
        language_source=args.language_source,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    env = make_env(calvin_root=args.calvin_root, scene=args.scene, show_gui=args.show_gui, use_egl=args.use_egl)
    results: list[int] = []
    try:
        for initial_state, eval_sequence in tqdm(selected, desc="calvin-rollout"):
            result = evaluate_sequence(
                env=env,
                model=model,
                task_oracle=task_oracle,
                initial_state=initial_state,
                eval_sequence=eval_sequence,
                val_annotations=val_annotations,
                ep_len=args.ep_len,
            )
            results.append(result)
    finally:
        env.close()
        env.ownsPhysicsClient = False
        env.cid = -1

    chain_success = {str(i + 1): sr for i, sr in enumerate(count_success(results))}
    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "calvin_root": str(Path(args.calvin_root).expanduser().resolve()),
        "scene": args.scene,
        "device": str(device),
        "num_sequences": len(selected),
        "start_sequence": args.start_sequence,
        "ep_len": args.ep_len,
        "language_source": args.language_source,
        "language_embedding_backend": args.language_embedding_backend,
        "task_embeddings_path": str(Path(args.task_embeddings_path).expanduser().resolve()) if args.task_embeddings_path else None,
        "language_embedding_dim": model.language_embedding_dim,
        "language_conditioned": model.language_embedding_dim > 0,
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "results": results,
        "avg_seq_len": float(np.mean(results)),
        "chain_success": chain_success,
        "task_info": get_task_stats(results, selected),
        "metric_type": "closed_loop_calvin_success_rate",
        "notes": (
            f"ACT policy consumes CALVIN language goals via {args.language_embedding_backend} state features."
            if model.language_embedding_dim > 0
            else "ACT policy is vision/state-only; CALVIN language annotation is passed through the official interface but not consumed by this policy."
        ),
    }
    json_dump(payload, args.output_json)
    print(json.dumps(payload["chain_success"], indent=2, sort_keys=True))
    print(f"avg_seq_len={payload['avg_seq_len']:.4f}")


if __name__ == "__main__":
    main()
