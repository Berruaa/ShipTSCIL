from methods.linear_probe import LinearProbeMethod
from methods.cil_naive import CILNaiveMethod
from methods.replay_dummy import ReplayDummyMethod
from methods.svm import SVMMethod


def build_method(method_name, model_name, num_classes, train_dataset, device, lr):
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

    if method_name in {"raw_replay", "latent_replay", "replay_dummy"}:
        return ReplayDummyMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    raise ValueError(f"Unknown method: {method_name}")