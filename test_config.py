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
device = "cuda:0"

# Model configure
model_arch_name = "vgg11"## undefined
model_num_classes = 1000## undefined

# Experiment name, easy to save weights and log files
exp_name = "VGG19-ImageNet_1K"## undefined

# Dataset address
test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

dataset_mean_normalize = (0.485, 0.456, 0.406)## undefined
dataset_std_normalize = (0.229, 0.224, 0.225)## undefined

resized_image_size = 256## undefined
crop_image_size = 224## undefined
batch_size = 256## undefined
num_workers = 4## undefined

# How many iterations to print the testing result
test_print_frequency = 20

model_weights_path = "./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar"
