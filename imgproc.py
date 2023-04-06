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
import random
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import Resize, ConvertImageDtype, Normalize
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image", "preprocess_one_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor: ## defines a function which converts an image to a tensor
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image) ## uses the to_tensor method to transform the image to a tensor

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm: ## checks if a range norm was provided
        tensor = tensor.mul(2.0).sub(1.0) ## tansforms the tensor by miltiplying it with 2 and substracting 1 from it

    # Convert torch.float32 image data type to torch.half image data type
    if half: ## checks if the half option was provided 
        tensor = tensor.half() ## changes the datatype of the tensor to f16

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any: ## defines a function which convert a tensor to an image
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:
        tensor = tensor.add(1.0).div(2.0) ## transforms the tensor by adding 1 to it and dividing it by 2

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half() ## changes the datatype of the tensor to f16

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8") ## converts the tensor to the image by applying multiple transformations

    return image


def preprocess_one_image( ## defines a method which preprocess an image
        image_path: str,
        image_size: int,
        range_norm: bool,
        half: bool,
        mean_normalize: tuple,
        std_normalize: tuple,
        device: torch.device,
) -> torch.Tensor:
    image = cv2.imread(image_path) ## load the image using OpenCV

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## converts te image from BGR to RGB

    # OpenCV convert PIL
    image = Image.fromarray(image) ## converts from OpenCV image to PIL image

    # Resize
    image = Resize([image_size, image_size])(image) ## resizes the image with the provided image_size
    # Convert image data to pytorch format data
    tensor = image_to_tensor(image, range_norm, half).unsqueeze_(0)
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(mean_normalize, std_normalize)(tensor)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device, non_blocking=True)

    return tensor


def center_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor], ## undefined
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]
    else:
        image_height, image_width = images[0].shape[0:2]

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2
    left = (image_width - patch_size) // 2

    # Crop lr image patch
    if input_type == "Tensor": ## checks if the input_type is a Tensor
        images = [image[## undefined
                  :,## undefined
                  :,## undefined
                  top:top + patch_size,## undefined
                  left:left + patch_size] for image in images]## undefined
    else:
        images = [image[
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:## checks if there is only one image
        images = images[0]

    return images


def random_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## sets the input type by checking if the provided images are tensors

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]## undefined
    else:
        image_height, image_width = images[0].shape[0:2]## undefined

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size)## undefined
    left = random.randint(0, image_width - patch_size)## undefined

    # Crop lr image patch
    if input_type == "Tensor":## undefined
        images = [image[## undefined
                  :,## undefined
                  :,## undefined
                  top:top + patch_size,## undefined
                  left:left + patch_size] for image in images]## undefined
    else:
        images = [image[
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_rotate(## undefined
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,## undefined
        center: tuple = None,## undefined
        rotate_scale_factor: float = 1.0## undefined
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Random select specific angle
    angle = random.choice(angles)## undefined

    if not isinstance(images, list):## undefined
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]## undefined
    else:
        image_height, image_width = images[0].shape[0:2]## undefined

    # Rotate LR image
    if center is None:
        center = (image_width // 2, image_height // 2)## undefined

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)## undefined

    if input_type == "Tensor":
        images = [F_vision.rotate(image, angle, center=center) for image in images]## undefined
    else:
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images]## undefined

    # When image number is 1
    if len(images) == 1:## undefined
        images = images[0]## undefined

    return images


def random_horizontally_flip(## undefined
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get horizontal flip probability
    flip_prob = random.random()## undefined

    if not isinstance(images, list):## undefined
        images = [images]## undefined

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## undefined

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.hflip(image) for image in images]## undefined
        else:
            images = [cv2.flip(image, 1) for image in images]## undefined

    # When image number is 1
    if len(images) == 1:## undefined
        images = images[0]## undefined

    return images## undefined


def random_vertically_flip(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get vertical flip probability
    flip_prob = random.random()## undefined

    if not isinstance(images, list):## undefined
        images = [images]## undefined

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.vflip(image) for image in images]## undefined
        else:
            images = [cv2.flip(image, 0) for image in images]## undefined

    # When image number is 1
    if len(images) == 1:## undefined
        images = images[0]## undefined

    return images
