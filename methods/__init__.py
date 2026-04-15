from methods.linear_probe import LinearProbeMethod
from methods.cil_naive import CILNaiveMethod
from methods.cil_replay_raw import CILReplayRawMethod
from methods.cil_replay_latent import CILReplayLatentMethod
from methods.cil_lwf import CILLwFMethod
from methods.cil_ncm import CILNCMMethod
from methods.cil_herding_ncm import CILHerdingNCMMethod
from methods.svm import SVMMethod

STANDARD_METHODS = {"linear_probe", "svm"}
SEQUENTIAL_METHODS = {"cil_naive", "cil_replay_raw", "cil_replay_latent", "cil_lwf", "cil_ncm", "cil_herding_ncm"}
REPLAY_METHODS = {"cil_replay_raw", "cil_replay_latent"}
DISTILLATION_METHODS = {"cil_lwf", "cil_replay_latent"}

_LORA_INCOMPATIBLE = {
    "cil_replay_latent", "latent_replay",
    "svm",
    "cil_ncm", "ncm",
    "cil_herding_ncm", "herding_ncm",
}


def build_method(
    method_name,
    model_name,
    num_classes,
    train_dataset,
    device,
    lr,
    replay_buffer_size=1000,
    replay_batch_size=32,
    balanced_replay=True,
    balanced_loss=True,
    use_distillation=False,
    distill_temperature=2.0,
    distill_weight=1.0,
    herding_replay=False,
    lora_config=None,
):
    if lora_config and lora_config.get("enabled") and method_name in _LORA_INCOMPATIBLE:
        raise ValueError(
            f"LoRA is incompatible with method '{method_name}'. "
            f"LoRA requires gradient-based training with raw time-series data. "
            f"Compatible methods: linear_probe, cil_naive, cil_replay_raw, cil_lwf."
        )

    if method_name == "linear_probe":
        return LinearProbeMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            lora_config=lora_config,
        )

    if method_name == "svm":
        return SVMMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    if method_name == "cil_naive":
        return CILNaiveMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            lora_config=lora_config,
        )

    if method_name in {"cil_replay_raw", "raw_replay"}:
        return CILReplayRawMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            replay_buffer_size=replay_buffer_size,
            replay_batch_size=replay_batch_size,
            balanced_replay=balanced_replay,
            balanced_loss=balanced_loss,
            herding_replay=herding_replay,
            lora_config=lora_config,
        )

    if method_name in {"cil_replay_latent", "latent_replay"}:
        return CILReplayLatentMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            replay_buffer_size=replay_buffer_size,
            replay_batch_size=replay_batch_size,
            balanced_replay=balanced_replay,
            balanced_loss=balanced_loss,
            use_distillation=use_distillation,
            distill_temperature=distill_temperature,
            distill_weight=distill_weight,
            herding_replay=herding_replay,
            lora_config=lora_config,
        )

    if method_name in {"cil_lwf", "lwf"}:
        return CILLwFMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            distill_temperature=distill_temperature,
            distill_weight=distill_weight,
            balanced_loss=balanced_loss,
            lora_config=lora_config,
        )

    if method_name in {"cil_ncm", "ncm"}:
        if herding_replay:
            return CILHerdingNCMMethod(
                model_name=model_name,
                num_classes=num_classes,
                train_dataset=train_dataset,
                device=device,
                lr=lr,
                replay_buffer_size=replay_buffer_size,
            )
        return CILNCMMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    if method_name in {"cil_herding_ncm", "herding_ncm"}:
        return CILHerdingNCMMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            replay_buffer_size=replay_buffer_size,
        )

    raise ValueError(f"Unknown method: {method_name}")