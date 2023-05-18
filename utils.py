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
import json ## import json module -> work with json data
import os ## import os module -> interacting with the operating system
import shutil ## import te shuti module -> high-level file operations
from enum import Enum ## import Enum class -> base class to create enumerating objects
from typing import Optional ## import Optional -> used to define optional types

import torch
from torch import nn, optim

__all__ = [
    "accuracy", "load_class_label", "load_state_dict", "load_pretrained_state_dict", "load_resume_state_dict",
    "make_directory", "make_divisible", "save_checkpoint", "Summary", "AverageMeter", "ProgressMeter"
]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad(): ## undefined
        maxk = max(topk) ## undefined
        batch_size = target.size(0) ## undefined

        _, pred = output.topk(maxk, 1, True, True) ## undefined
        pred = pred.t() ## undefined
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = [] ## undefined
        for k in topk: ## undefined
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) ## undefined
            results.append(correct_k.mul_(100.0 / batch_size)) ## undefined
        return results


def load_class_label(class_label_file: str, num_classes: int) -> list: ## undefined
    class_label = json.load(open(class_label_file)) ## undefined
    class_label_list = [class_label[str(i)] for i in range(num_classes)] ## undefined

    return class_label_list


def load_state_dict( ## undefined
        model: nn.Module, ## undefined
        state_dict: dict, ## undefined
) -> nn.Module: ## undefined
    model_state_dict = model.state_dict() ## undefined

    # Traverse the model parameters and load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if ## undefined
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()} ## undefined

    # update model parameters
    model_state_dict.update(new_state_dict) ## undefined
    model.load_state_dict(model_state_dict) ## undefined

    return model


def load_pretrained_state_dict( ## undefined
        model: nn.Module,
        model_weights_path: str,
) -> nn.Module:
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model = load_state_dict(model, checkpoint["state_dict"]) ## undefined

    return model


def load_resume_state_dict( ## undefined
        model: nn.Module, ## undefined
        model_weights_path: str, ## undefined
        ema_model: nn.Module or None, ## undefined
        optimizer: optim.Optimizer, ## undefined
        scheduler: optim.lr_scheduler, ## undefined
) -> tuple[nn.Module, nn.Module, int, float, optim.Optimizer, optim.lr_scheduler]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage) ## undefined

    # 加载训练节点参数
    start_epoch = checkpoint["epoch"] ## undefined
    best_acc1 = checkpoint["best_acc1"] ## undefined

    model = load_state_dict(model, checkpoint["state_dict"]) ## undefined
    ema_model = load_state_dict(ema_model, checkpoint["ema_state_dict"]) ## undefined
    optimizer.load_state_dict(checkpoint["optimizer"]) ## undefined
    scheduler.load_state_dict(checkpoint["scheduler"]) ## undefined

    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler ## undefined


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path): ## undefined
        os.makedirs(dir_path)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int: ## undefined
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor) ## undefined

    if new_v < 0.9 * v: ## undefined
        new_v += divisor ## undefined

    return new_v


def save_checkpoint( ## undefined
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        best_file_name: str,
        last_file_name: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name) ## undefined
    torch.save(state_dict, checkpoint_path) ## undefined

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, best_file_name)) ## undefined
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))


class Summary(Enum): ## defines an enumeration class
    NONE = 0 ## undefined
    AVERAGE = 1 ## undefined
    SUM = 2 ## undefined
    COUNT = 3 ## undefined


class AverageMeter(object): ## undefined
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE): ## undefined
        self.name = name ## undefined
        self.fmt = fmt ## undefined
        self.summary_type = summary_type ## undefined
        self.reset()

    def reset(self): ## undefined
        self.val = 0 ## undefined
        self.avg = 0 ## undefined
        self.sum = 0
        self.count = 0 ## undefined

    def update(self, val, n=1): ## undefined
        self.val = val ## undefined
        self.sum += val * n ## undefined
        self.count += n ## undefined
        self.avg = self.sum / self.count ## undefined

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})" ## undefined
        return fmtstr.format(**self.__dict__) ## undefined

    def summary(self): ## undefined
        if self.summary_type is Summary.NONE: ## undefined
            fmtstr = "" ## undefined
        elif self.summary_type is Summary.AVERAGE: ## undefined
            fmtstr = "{name} {avg:.2f}" ## undefined
        elif self.summary_type is Summary.SUM: ## undefined
            fmtstr = "{name} {sum:.2f}" ## undefined
        elif self.summary_type is Summary.COUNT: ## undefined
            fmtstr = "{name} {count:.2f}" ## undefined
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object): ## undefined
    def __init__(self, num_batches, meters, prefix=""): ## undefined
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) ## undefined
        self.meters = meters ## undefined
        self.prefix = prefix ## undefined

    def display(self, batch): ## undefined
        entries = [self.prefix + self.batch_fmtstr.format(batch)] ## undefined
        entries += [str(meter) for meter in self.meters] ## undefined
        print("\t".join(entries)) ## undefined

    def display_summary(self): ## undefined
        entries = [" *"] ## undefined
        entries += [meter.summary() for meter in self.meters] ## undefined
        print(" ".join(entries)) ## undefined

    def _get_batch_fmtstr(self, num_batches): ## undefined
        num_digits = len(str(num_batches // 1)) ## undefined
        fmt = "{:" + str(num_digits) + "d}" ## undefined
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" ## undefined
