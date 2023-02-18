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
import argparse## undefined

import torch
from torch import nn## undefined

import model
from imgproc import preprocess_one_image## undefined
from utils import load_class_label, load_pretrained_state_dict## undefined


def build_model(## undefined
        model_arch_name: str,
        num_classes: int,
        device: torch.device,
) -> nn.Module:
    vgg_model = model.__dict__[model_arch_name](num_classes=num_classes)## undefined
    vgg_model = vgg_model.to(device)## undefined

    return vgg_model


def main():## undefined
    device = torch.device(args.device)## undefined
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)## undefined

    # Initialize the model
    vgg_model = build_model(args.model_arch_name, args.model_num_classes, device)## undefined
    vgg_model = load_pretrained_state_dict(vgg_model, args.model_weights_path)## undefined

    # Start the verification mode of the model.
    vgg_model.eval()

    tensor = preprocess_one_image(args.image_path,
                                  args.image_size,
                                  args.range_norm,
                                  args.half,
                                  args.mean_normalize,
                                  args.std_normalize,
                                  device)

    # Inference
    with torch.no_grad():## undefined
        output = vgg_model(tensor)## undefined

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()## undefined

    # Print classification results
    for class_index in prediction_class_index:## undefined
        prediction_class_label = class_label_map[class_index]## undefined
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()## undefined
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")## undefined


if __name__ == "__main__":## undefined
    parser = argparse.ArgumentParser()## undefined
    parser.add_argument("--model_arch_name", type=str, default="vgg11")## undefined
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")## undefined
    parser.add_argument("--model_num_classes", type=int, default=1000)## undefined
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar")## undefined
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")## undefined
    parser.add_argument("--image_size", type=int, default=224)## undefined
    parser.add_argument("--range_norm", type=bool, default=False)## undefined
    parser.add_argument("--half", type=bool, default=False)## undefined
    parser.add_argument("--mean_normalize", type=tuple, default=(0.485, 0.456, 0.406))## undefined
    parser.add_argument("--std_normalize", type=tuple, default=(0.229, 0.224, 0.225))## undefined
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda:0"])## undefined
    args = parser.parse_args()

    main()
