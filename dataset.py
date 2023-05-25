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
import queue ## module for queue data structures
import sys ## module for system functions
import threading ## module for threaded programming
from glob import glob ## uses regex to search for system paths using specified rules

import cv2 ## imports open cv for image processing
import torch
from PIL import Image ## imports Image data type from PIL for opening,creating,manipulating images
from torch.utils.data import Dataset, DataLoader ## Dataset are the actual datasets. DataLoader makes them iterable
from torchvision import transforms ## transforms includes methods for transformations such as grayscale,padding,linear transformations
from torchvision.datasets.folder import find_classes ## find_classes finds classes in a dataset and returns them as a structured tree
from torchvision.transforms import TrivialAugmentWide ## module that has methods for augmenting(?) tensors 

from imgproc import image_to_tensor

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp") ## list of image extensions supported

# The delimiter is not the same between different platforms
if sys.platform == "win32": ## delimiter for system paths
    delimiter = "\\" ## for windows, this is \\
else:
    delimiter = "/" ## otherwise, it's / (as it should be)


class ImageDataset(Dataset): ## class for handling image datasets
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
            images_dir: str, ## location of the images
            resized_image_size: int, ## does nothing. suggests that the parameter should be an int
            crop_image_size: int, ## does nothing. suggests that the parameter should be an int 
            mean_normalize: tuple = None, ## does nothing. suggests that the parameter should be a tuple and the default value is None
            std_normalize: tuple = None, ## does nothing. suggests that the parameter should be a tuple and the default value is None
            mode: str = "train", ## sets the mode to traning (since this is hardcoded to train)
    ) -> None:
        super(ImageDataset, self).__init__() ## calls the constructor of the parent class (Dataset) with ImageDataset as a parameter
        if mean_normalize is None:
            mean_normalize = (0.485, 0.456, 0.406) ## if no value is provided, sets the mentioned values
        if std_normalize is None:
            std_normalize = (0.229, 0.224, 0.225) ## if no value is provided, sets the mentioned values
        # Iterate over all image paths
        self.images_file_path = glob(f"{images_dir}/*/*") ## makes images_file_path a list with all paths that start with images_dir and have depth 2
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(images_dir) ## puts a dictionary of form foldername:index containing all folders in the dataset in the class_to_idx variable
        self.crop_image_size = crop_image_size ## sets the self parameter as the one passed in the constructor
        self.resized_image_size = resized_image_size ## sets the self parameter as the one passed in the constructor
        self.mean_normalize = mean_normalize ## sets the self parameter as the one passed in the constructor
        self.std_normalize = std_normalize ## sets the self parameter as the one passed in the constructor
        self.mode = mode ## sets the self parameter as the one passed in the constructor
        self.delimiter = delimiter

        if self.mode == "Train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## this line composes all the transforms between []
                transforms.RandomResizedCrop((self.resized_image_size, self.resized_image_size)), ## crops a random portion of the image to the mentioned sizes
                TrivialAugmentWide(), ## unsure
                transforms.RandomRotation([0, 270]), ## rotates the image by a random no of degrees between 0 and 270
                transforms.RandomHorizontalFlip(0.5), ## chance of 50% to flip horizontally
                transforms.RandomVerticalFlip(0.5), ## change of 50% to flip vertically
            ])
        elif self.mode == "Valid" or self.mode == "Test": ## defines transformations for test mode
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## composes the transforms between []
                transforms.Resize((self.resized_image_size, self.resized_image_size)), ## resizes the image to the given sizes
                transforms.CenterCrop((self.crop_image_size, self.crop_image_size)), ## only keeps a certain size out from the center of the image
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([ ## defines post transformations (?i think these are applied regardless of what is applied before them)
            transforms.ConvertImageDtype(torch.float), ## converts from PIL to numpy array of floats
            transforms.Normalize(self.mean_normalize, self.std_normalize) ## normalizes the input data
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        images_dir, images_name = self.images_file_path[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if images_name.split(".")[-1].lower() in IMG_EXTENSIONS: ## checks if the extension of the file is in the IMG_EXTENSIONS array
            image = cv2.imread(self.images_file_path[batch_index]) ## opens the image
            target = self.class_to_idx[images_dir] ## gets the index of the taget folder 
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, " ## throws error if incompatible image extension
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## converts image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) ## converts image from PIL to Image

        # Data preprocess
        image = self.pre_transform(image) ## makes data easier for the model to process

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = image_to_tensor(image, False, False) ## converts from Image to tensor

        # Data postprocess
        tensor = self.post_transform(tensor) ## not sure what exactly this does, transforms model output to more suitable format (?)

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.images_file_path)


class PrefetchGenerator(threading.Thread): ## used to import data in parallel to the model working on other data
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self) ## spawns a thread
        self.queue = queue.Queue(num_data_prefetch_queue) ## creates a queue with a capacity of num_data_prefetch_queue
        self.generator = generator ## assigns the value from the constructor to the generator value of this class
        self.daemon = True ## sets the thread as a daemon - it will be killed if the parent thread is killed
        self.start() ## starts the thread

    def run(self) -> None: ## executes when thread is spawned
        for item in self.generator: ## iterates through the elements in the data generator
            self.queue.put(item) ## adds them to the queue
        self.queue.put(None) ## adds a null element after it finishes adding all elements

    def __next__(self): ## returns the next element of the queue
        next_item = self.queue.get() ## gets the next item from the queue
        if next_item is None: ## checks if queue is empty by using the null element added above
            raise StopIteration ## if empty, stops iterating
        return next_item

    def __iter__(self): ## returns instance of this class
        return self


class PrefetchDataLoader(DataLoader): ## prefetch data loader to load the data for prefetching
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues. ## how many queues for pre loading
        kwargs (dict): Other extended parameters. ## the rest of the parameters
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue ## assignes the provided value to the variable
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue) ## returns the iterator object, built by inheriting DataLoader


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None: ## the initializer function for the CPUPrefetcher class
        self.original_dataloader = dataloader ## original_dataloader is initialized with the dataloader passed as a parameter
        self.data = iter(dataloader) ## the data is initialized to an iterator that iterates over the data in the dataloader

    def next(self): ## method for returning the next item from the iterator
        try:
            return next(self.data) ## returns the next item from the iterator
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader) ## returns the nr of items 


class CUDAPrefetcher: ## prefetcher class for improved performance
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ## initializer function for this class
        self.batch_data = None ## batch data initialised to empty
        self.original_dataloader = dataloader ## original_dataloader is initialized to the dataloader in the parameter
        self.device = device ## device parameter of the class is initialized to the passed device parameter

        self.data = iter(dataloader) ## data is initialized to an iterator with the passed data
        self.stream = torch.cuda.Stream() ## stream is initilalized to an empty stream of type cuda
        self.preload() ## data is preloaded using the function defined below

    def preload(self): ## function for preloading data
        try:
            self.batch_data = next(self.data) ## get next chunk of data
        except StopIteration: ## if data is finished
            self.batch_data = None ## set data to none since there is no more data
            return None

        with torch.cuda.stream(self.stream): ## sets a stream of type CUDA
            for k, v in self.batch_data.items(): ## iterates over the items in batch_data
                if torch.is_tensor(v): ## if v is tensor
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) ## moves tensor to specified device

    def next(self): ## function for getting  next batch of data
        torch.cuda.current_stream().wait_stream(self.stream) ## waits for the current stream to finish
        batch_data = self.batch_data ## assigns the current batch data to the variable
        self.preload() ## preloads the next batch
        return batch_data

    def reset(self): ## resets the stream to the beginning
        self.data = iter(self.original_dataloader) ## creates a new iterator from the original data loader
        self.preload() ## loads the data

    def __len__(self) -> int:
        return len(self.original_dataloader)
