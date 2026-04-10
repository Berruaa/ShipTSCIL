from methods.linear_probe import LinearProbeMethod
from methods.cil_naive import CILNaiveMethod
from methods.cil_replay_raw import CILReplayRawMethod
from methods.cil_replay_latent import CILReplayLatentMethod
from methods.svm import SVMMethod


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
):
    if method_name == "linear_probe":
        return LinearProbeMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
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
        )

    raise ValueError(f"Unknown method: {method_name}")