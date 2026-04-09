from methods.linear_probe import LinearProbeMethod
from methods.cil_naive import CILNaiveMethod
from methods.cil_replay_raw import CILReplayRawMethod
from methods.replay_dummy import ReplayDummyMethod
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
        )

    if method_name in {"cil_replay_latent", "latent_replay", "replay_dummy"}:
        return ReplayDummyMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    raise ValueError(f"Unknown method: {method_name}")