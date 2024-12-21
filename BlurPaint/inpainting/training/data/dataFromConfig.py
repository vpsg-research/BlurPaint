import pytorch_lightning as pl
import torch
import os
import glob
import sys
import albumentations as A
import numpy as np

from PIL import Image
from functools import partial
from torch.utils.data import Dataset, DataLoader

from inpainting.training.data.utils import Txt2ImgIterableBaseDataset
# from inpainting.training.data.masks import get_mask_generator

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_edge(image, sigma):

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))

    return edge, gray_image


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

def make_train_dataloader(img_dir, mask_dir=None, out_size=256,
                          img_transform='resize', mask_transform='resize',
                          use_masks=True):
    if not use_masks:
        mask_transform = None
    else:
        mask_transform = get_transforms(mask_transform, out_size)
    transform = get_transforms(img_transform, out_size)
    dataset = WrappedDataset(img_dir=img_dir, img_transform=transform, mask_dir=mask_dir,
                             mask_transform=mask_transform, use_masks=use_masks)

    return dataset

def get_transforms(transform_variant, out_size):
        transform = None
        if transform_variant == 'default':
            transform = A.Compose([
                A.RandomScale(scale_limit=0.2),  # +/- 20%
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            ])
        elif transform_variant == 'resize':
            transform = A.Resize(height=out_size, width=out_size)
        elif transform_variant == 'resize_bed':
            transform = A.Compose([
                A.RandomScale(scale_limit=0.2),  # +/- 20%
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
            ])
        return transform

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, img_dir, img_transform, mask_dir=None, mask_transform=None, use_masks=False):
        self.use_masks = use_masks
        if self.use_masks:
            self.mask_data = list(glob.glob(os.path.join(sys.path[0], mask_dir, '*.png'), recursive=True))
            self.mask_transform = mask_transform
            self.mask_data_len = len(self.mask_data)

        self.img_data = sorted(list(glob.glob(os.path.join(sys.path[0], img_dir, '*.jpg'), recursive=True)))
        self.img_transform = img_transform

        self.device = torch.device("cuda" if torch.cuda.is_available() else "")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data[idx]
        image = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)
        image = self.img_transform(image=image)['image']
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1) # h,w,c  -> c,h,w
        image = torch.from_numpy(image)

        if self.use_masks:
            mask_path = self.mask_data[idx % self.mask_data_len]
            mask = np.array(Image.open(mask_path).convert("L")).astype(np.uint8)
            mask = self.mask_transform(image=mask)['image']
            mask = mask.astype(np.float32) / 255.0
            mask = mask[None]
            mask = torch.from_numpy(mask)

            masked_image = (1 - mask) * image

            batch = {"image": image, "mask": mask, "masked_image": masked_image}
        else:
            batch = {"image": image}

        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0
        return batch



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def setup(self, stage=None):
        self.datasets = dict(
            (k, self.dataset_configs[k])
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = make_train_dataloader(**self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle,
                          persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)