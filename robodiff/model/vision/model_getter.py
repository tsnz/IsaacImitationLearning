import torch
import torch.nn as nn
import torchvision


def get_resnet(name, input_channel=3, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    if input_channel != 3:
        resnet.conv1 = nn.Conv2d(
            input_channel, resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model
