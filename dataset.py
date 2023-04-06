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
import torch ## includes the toch library -> framework for building deep learning models
from PIL import Image ## includes the Image class from the Python Imaging Library -> adds image processing capabilities
from torch.utils.data import Dataset, DataLoader ## includes the Dataset and Dataloader classes from torch
from torchvision import transforms ## includes the transforms module from torch -> common image transforms
from torchvision.datasets.folder import find_classes ## includes the find_classes function from torch -> Finds the class folders in a dataset
from torchvision.transforms import TrivialAugmentWide ## includes the TrivialAugmentWide class from torch

from imgproc import image_to_tensor 

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp") ## defines different kind of image extensions

# The delimiter is not the same between different platforms
if sys.platform == "win32": ## checks if the current os on this device is windows
    delimiter = "\\" ## the directory delimiter in windows is //
else:
    delimiter = "/" ## the directory delimiter in linux is /


class ImageDataset(Dataset): ## define a class dataset for images being a child class of Dataset
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
            images_dir: str, ## the path to the directory where the images are stores
            resized_image_size: int, ## the size of the resized image
            crop_image_size: int, ## the size of the cropped image
            mean_normalize: tuple = None, ## undefined
            std_normalize: tuple = None, ## undefined
            mode: str = "train", ## the mode of the dataset -> it is train by default
    ) -> None:
        super(ImageDataset, self).__init__() ## calls the constructor of the Dataset class
        if mean_normalize is None:
            mean_normalize = (0.485, 0.456, 0.406) ## sets the mean normalize vector if none was provided
        if std_normalize is None:
            std_normalize = (0.229, 0.224, 0.225) ## sets the standard normalize vector if none was provided
        # Iterate over all image paths
        self.images_file_path = glob(f"{images_dir}/*/*") ## gets the path of the directory where the images are stored
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(images_dir) ## 
        self.crop_image_size = crop_image_size ## initializes the crop_image_size of the class
        self.resized_image_size = resized_image_size ## initializes the resized_image_size of the class
        self.mean_normalize = mean_normalize ##  initializes the mean normalize vector of the class
        self.std_normalize = std_normalize ## initializes the standard normalize vector of the class
        self.mode = mode ## initializes the mode of the Dataset
        self.delimiter = delimiter

        if self.mode == "Train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## computes the initial transformation by composing several transformations
                transforms.RandomResizedCrop((self.resized_image_size, self.resized_image_size)), ## Crop a random portion of image and resize it to the provided resized image size
                TrivialAugmentWide(), ## undefined
                transforms.RandomRotation([0, 270]), ## performs a random rotation between 0 and 270 degrees
                transforms.RandomHorizontalFlip(0.5), ## flips the image horizontally with a probability of 0.5
                transforms.RandomVerticalFlip(0.5), ## flips the image vertically with a probability of 0.5
            ])
        elif self.mode == "Valid" or self.mode == "Test": ## checks if the mode of the Dataset is either Valid or Test
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## computes the initial transformation by composing several transformations
                transforms.Resize((self.resized_image_size, self.resized_image_size)), ## resizes the image to the provided resized image size
                transforms.CenterCrop((self.crop_image_size, self.crop_image_size)), ## crops the image in the center with the provided crop image size
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([ ## compute the post transformation of the dataset by composing multiple transformations
            transforms.ConvertImageDtype(torch.float), ## Converts the image data type to float
            transforms.Normalize(self.mean_normalize, self.std_normalize) ## normalizes the image with the provided mean and std normalize vectors
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        images_dir, images_name = self.images_file_path[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if images_name.split(".")[-1].lower() in IMG_EXTENSIONS: ## checks for the files which have the image extension
            image = cv2.imread(self.images_file_path[batch_index]) ## use the OpenCV library to load the image
            target = self.class_to_idx[images_dir] ## undefined
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, " ## undefined
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## uses the OpenCV library to convert the image from BGR ro RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) ## Converts the OpenCV image to PIL image

        # Data preprocess
        image = self.pre_transform(image) ## applies the pre transformation

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = image_to_tensor(image, False, False) ## convert the image data into Tensor

        # Data postprocess
        tensor = self.post_transform(tensor) ## applies the post transformation to the tensor

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.images_file_path)


class PrefetchGenerator(threading.Thread): ## defines a class inheriting from the Thread class
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self) ## calls the super class contructor
        self.queue = queue.Queue(num_data_prefetch_queue) ## add a queue to the class
        self.generator = generator ## sets the generator to the provided one
        self.daemon = True ## sets daemon to true
        self.start() ## starts the thread

    def run(self) -> None: ## the run method of the thread class
        for item in self.generator: ## iterates through each element in the generator
            self.queue.put(item) ## puts each element in the queue
        self.queue.put(None) ## add a None element in the queue

    def __next__(self): ## a method used for to get the next item for the iterator
        next_item = self.queue.get() ## get the next item in the queue
        if next_item is None: ## checks if we got all items
            raise StopIteration ## stops the iteration
        return next_item

    def __iter__(self): ## returns the iterator object itself. If required
        return self


class PrefetchDataLoader(DataLoader): ## defines a class inheriting from the DataLoader class
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues. ## undefined
        kwargs (dict): Other extended parameters. ## undefined
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue ## sets the num
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue) ## undefined


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None: ## the constructor of the class, requires a dataloader
        self.original_dataloader = dataloader ##  sets the dataloader to the provided one
        self.data = iter(dataloader) ## gets the iterator of the dataloader

    def next(self): ## a method used to iterate through data
        try:
            return next(self.data) ## get the next element from the data
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader) ## returns the length of the provided dataloader


class CUDAPrefetcher: ## defines a class called CUDAPrefetcher
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ## the constructor of the class, which takes as parameters the dataloader and the device 
        self.batch_data = None ## initialize the batch_data to None
        self.original_dataloader = dataloader ## initialize the dataloader with the provided one
        self.device = device ## initialize the device with the provided one

        self.data = iter(dataloader) ## initialize the data as the iterator of the dataloader
        self.stream = torch.cuda.Stream() ## initialize the stream as a CUDA Stream
        self.preload() ## calls the preload method 

    def preload(self): ## defines the preload method
        try:
            self.batch_data = next(self.data) ## get the next element from the data
        except StopIteration: ## checks if the iterator ended
            self.batch_data = None ## sets the batch_data to None
            return None

        with torch.cuda.stream(self.stream): ## uses the CUDA Stream
            for k, v in self.batch_data.items(): ## gets the item from the batch_data
                if torch.is_tensor(v): ## checks if the provided value is tensor
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) ## undefined

    def next(self): ## get the next item in the iteration
        torch.cuda.current_stream().wait_stream(self.stream) ## undefined
        batch_data = self.batch_data ## sets the batch_data with the current batch_data stored in the class
        self.preload() ## calls the preload method
        return batch_data

    def reset(self): ## a method to reset the data
        self.data = iter(self.original_dataloader) ## reinitialize the data as an iterator over the original_dataloader
        self.preload() ## calls the preload method

    def __len__(self) -> int:
        return len(self.original_dataloader)
