from methods.linear_probe import LinearProbeMethod
from methods.replay_dummy import ReplayDummyMethod


def build_method(method_name, model_name, num_classes, train_dataset, device, lr):
    if method_name == "linear_probe":
        return LinearProbeMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    if method_name == "replay_dummy":
        return ReplayDummyMethod(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

    raise ValueError(f"Unknown method: {method_name}")