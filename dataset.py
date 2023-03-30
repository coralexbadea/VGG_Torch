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
import queue ## includes the python's build-in queue module
import sys ## includes the python's build-in sys module
import threading ## includes the python's build-in sys module
from glob import glob ## includes the glob function from the glob module

import cv2 ## includes the opencv package
import torch
from PIL import Image ## includes the Image class from the Python Imaging Library
from torch.utils.data import Dataset, DataLoader ## includes the Dataset and Dataloader classes from torch
from torchvision import transforms ## includes the transforms module from torch
from torchvision.datasets.folder import find_classes ## includes the find_classes function from torch 
from torchvision.transforms import TrivialAugmentWide ## includes the TrivialAugmentWide class from torch

from imgproc import image_to_tensor

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp") ## defines different kind of image extensions

# The delimiter is not the same between different platforms
if sys.platform == "win32": ## undefined
    delimiter = "\\" ## undefined
else:
    delimiter = "/" ## undefined


class ImageDataset(Dataset): ## undefined
    """Define training/valid dataset loading methods.

    Args:
        images_dir (str): Train/Valid dataset address.
        resized_image_size (int): Resized image size.
        crop_image_size (int): Crop image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(
            self,
            images_dir: str, ## undefined
            resized_image_size: int, ## undefined
            crop_image_size: int, ## undefined
            mean_normalize: tuple = None, ## undefined
            std_normalize: tuple = None, ## undefined
            mode: str = "train", ## undefined
    ) -> None:
        super(ImageDataset, self).__init__() ## undefined
        if mean_normalize is None:
            mean_normalize = (0.485, 0.456, 0.406) ## undefined
        if std_normalize is None:
            std_normalize = (0.229, 0.224, 0.225) ## undefined
        # Iterate over all image paths
        self.images_file_path = glob(f"{images_dir}/*/*") ## undefined
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(images_dir) ## undefined
        self.crop_image_size = crop_image_size ## undefined
        self.resized_image_size = resized_image_size ## undefined
        self.mean_normalize = mean_normalize ## undefined
        self.std_normalize = std_normalize ## undefined
        self.mode = mode ## undefined
        self.delimiter = delimiter

        if self.mode == "Train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined
                transforms.RandomResizedCrop((self.resized_image_size, self.resized_image_size)), ## undefined
                TrivialAugmentWide(), ## undefined
                transforms.RandomRotation([0, 270]), ## undefined
                transforms.RandomHorizontalFlip(0.5), ## undefined
                transforms.RandomVerticalFlip(0.5), ## undefined
            ])
        elif self.mode == "Valid" or self.mode == "Test": ## undefined
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined
                transforms.Resize((self.resized_image_size, self.resized_image_size)), ## undefined
                transforms.CenterCrop((self.crop_image_size, self.crop_image_size)), ## undefined
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([ ## undefined
            transforms.ConvertImageDtype(torch.float), ## undefined
            transforms.Normalize(self.mean_normalize, self.std_normalize) ## undefined
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        images_dir, images_name = self.images_file_path[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if images_name.split(".")[-1].lower() in IMG_EXTENSIONS: ## undefined
            image = cv2.imread(self.images_file_path[batch_index]) ## undefined
            target = self.class_to_idx[images_dir] ## undefined
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, " ## undefined
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## undefined

        # OpenCV convert PIL
        image = Image.fromarray(image) ## undefined

        # Data preprocess
        image = self.pre_transform(image) ## undefined

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = image_to_tensor(image, False, False) ## undefined

        # Data postprocess
        tensor = self.post_transform(tensor) ## undefined

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.images_file_path)


class PrefetchGenerator(threading.Thread): ## undefined
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self) ## undefined
        self.queue = queue.Queue(num_data_prefetch_queue) ## undefined
        self.generator = generator ## undefined
        self.daemon = True ## undefined
        self.start() ## undefined

    def run(self) -> None: ## undefined
        for item in self.generator: ## undefined
            self.queue.put(item) ## undefined
        self.queue.put(None) ## undefined

    def __next__(self): ## undefined
        next_item = self.queue.get() ## undefined
        if next_item is None: ## undefined
            raise StopIteration ## undefined
        return next_item

    def __iter__(self): ## undefined
        return self


class PrefetchDataLoader(DataLoader): ## undefined
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues. ## undefined
        kwargs (dict): Other extended parameters. ## undefined
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue ## undefined
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue) ## undefined


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None: ## undefined
        self.original_dataloader = dataloader ## undefined
        self.data = iter(dataloader) ## undefined

    def next(self): ## undefined
        try:
            return next(self.data) ## undefined
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader) ## undefined


class CUDAPrefetcher: ## undefined
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ## undefined
        self.batch_data = None ## undefined
        self.original_dataloader = dataloader ## undefined
        self.device = device ## undefined

        self.data = iter(dataloader) ## undefined
        self.stream = torch.cuda.Stream() ## undefined
        self.preload() ## undefined

    def preload(self): ## undefined
        try:
            self.batch_data = next(self.data) ## undefined
        except StopIteration: ## undefined
            self.batch_data = None ## undefined
            return None

        with torch.cuda.stream(self.stream): ## undefined
            for k, v in self.batch_data.items(): ## undefined
                if torch.is_tensor(v): ## undefined
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) ## undefined

    def next(self): ## undefined
        torch.cuda.current_stream().wait_stream(self.stream) ## undefined
        batch_data = self.batch_data ## undefined
        self.preload() ## undefined
        return batch_data

    def reset(self): ## undefined
        self.data = iter(self.original_dataloader) ## undefined
        self.preload() ## undefined

    def __len__(self) -> int:
        return len(self.original_dataloader)
