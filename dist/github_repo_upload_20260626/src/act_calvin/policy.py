from __future__ import annotations

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act import ACTConfig, ACTPolicy


def make_act_config(
    *,
    image_size: int,
    chunk_size: int,
    n_action_steps: int | None = None,
    device: str = "cuda",
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    use_vae: bool = True,
    kl_weight: float = 10.0,
    lr: float = 1e-5,
    lr_backbone: float | None = None,
    weight_decay: float = 1e-4,
    temporal_ensemble_coeff: float | None = None,
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
    state_dim: int = 15,
) -> ACTConfig:
    """Create a LeRobot ACT config matching the CALVIN visual-action schema."""

    if n_action_steps is None:
        n_action_steps = chunk_size
    if lr_backbone is None:
        lr_backbone = lr

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.images.static": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.images.gripper": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        device=device,
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        use_vae=use_vae,
        kl_weight=kl_weight,
        optimizer_lr=lr,
        optimizer_lr_backbone=lr_backbone,
        optimizer_weight_decay=weight_decay,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        pretrained_backbone_weights=pretrained_backbone_weights,
    )


def make_policy(**kwargs) -> ACTPolicy:
    cfg = make_act_config(**kwargs)
    return ACTPolicy(cfg)
