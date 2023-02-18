# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import cast, Dict, List, Union ## undefined

import torch ## undefined
from torch import Tensor ## undefined
from torch import nn ## undefined

__all__ = [
    "VGG",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], ## undefined
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential: ## undefined
    layers: nn.Sequential[nn.Module] = nn.Sequential() ## undefined
    in_channels = 3 ## undefined
    for v in vgg_cfg: ## undefined
        if v == "M": ## undefined
            layers.append(nn.MaxPool2d((2, 2), (2, 2))) ## undefined
        else: ## undefined
            v = cast(int, v) ## undefined
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1)) ## undefined
            if batch_norm: ## undefined
                layers.append(conv2d) ## undefined
                layers.append(nn.BatchNorm2d(v)) ## undefined
                layers.append(nn.ReLU(True)) ## undefined
            else: ## undefined
                layers.append(conv2d) ## undefined
                layers.append(nn.ReLU(True)) ## undefined
            in_channels = v

    return layers


class VGG(nn.Module): ## undefined
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000) -> None: ## undefined
        super(VGG, self).__init__() ## undefined
        self.features = _make_layers(vgg_cfg, batch_norm) ## undefined

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) ## undefined

        self.classifier = nn.Sequential( ## undefined
            nn.Linear(512 * 7 * 7, 4096), ## undefined
            nn.ReLU(True), ## undefined
            nn.Dropout(0.5), ## undefined
            nn.Linear(4096, 4096), ## undefined
            nn.ReLU(True), ## undefined
            nn.Dropout(0.5), ## undefined
            nn.Linear(4096, num_classes), ## undefined
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x) ## undefined

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x) ## undefined
        out = self.avgpool(out) ## undefined
        out = torch.flatten(out, 1) ## undefined
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules(): ## undefined
            if isinstance(module, nn.Conv2d): ## undefined
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu") ## undefined
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0) ## undefined
            elif isinstance(module, nn.BatchNorm2d): ## undefined
                nn.init.constant_(module.weight, 1) ## undefined
                nn.init.constant_(module.bias, 0) ## undefined
            elif isinstance(module, nn.Linear): ## undefined
                nn.init.normal_(module.weight, 0, 0.01) ## undefined
                nn.init.constant_(module.bias, 0) ## undefined


def vgg11(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs) ## undefined

    return model


def vgg13(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)

    return model


def vgg19(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)

    return model


def vgg11_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], True, **kwargs)

    return model


def vgg13_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)

    return model


def vgg16_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)

    return model


def vgg19_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], True, **kwargs)

    return model
