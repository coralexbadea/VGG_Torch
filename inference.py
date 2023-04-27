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
import argparse## imports the python modules used to parse command line arguments

import torch
from torch import nn## imports the nn module of the torch library which is used to train and build the layers of neural networks such as input, hidden, and output

import model
from imgproc import preprocess_one_image## import the preprocess_one_image from the imgproc module
from utils import load_class_label, load_pretrained_state_dict## imports some functions from the utils module


def build_model(## defines a function which builds a model
        model_arch_name: str,
        num_classes: int,
        device: torch.device,
) -> nn.Module:
    vgg_model = model.__dict__[model_arch_name](num_classes=num_classes)## undefined
    vgg_model = vgg_model.to(device)## undefined

    return vgg_model


def main():## defines the main function of the module
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
    for class_index in prediction_class_index:## iterates through all the items inside the prediction_class_index list
        prediction_class_label = class_label_map[class_index]## gets the label of the current class_index
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()## undefined
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")## undefined


if __name__ == "__main__":## defines that this is the entry point of the program
    parser = argparse.ArgumentParser()## creates the command line parser
    parser.add_argument("--model_arch_name", type=str, default="vgg11")## adds the model architecture name as a command line argument with the default value of "vgg11"
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")## adds the class label file as a command line argument wit the default value of "./data/ImageNet_1K_labels_map.txt" 
    parser.add_argument("--model_num_classes", type=int, default=1000)## adds the numer of model classes as a command line argument with the default value of 1000
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar")## adds the path to the weigths of the model as a command line argument
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")## adds the path to the image as a command line argument
    parser.add_argument("--image_size", type=int, default=224)## adds the size of the image as a command line argument
    parser.add_argument("--range_norm", type=bool, default=False)## adds the range_norm(found in the imgproc.py as argument)as a command line argument with the default value of False
    parser.add_argument("--half", type=bool, default=False)## adds the half(found in the imgproc.py as argument) as a command line argument with the default value of False
    parser.add_argument("--mean_normalize", type=tuple, default=(0.485, 0.456, 0.406))## adds the mean normalize vector as a command line argument with the default value of (0.485, 0.456, 0.406) 
    parser.add_argument("--std_normalize", type=tuple, default=(0.229, 0.224, 0.225))## adds the standard normalize vector as a command line argument with the default value of (0.229, 0.224, 0.225)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda:0"])## adds the device as a command line argument with values "cpu" and "cuda:0"
    args = parser.parse_args()

    main()
