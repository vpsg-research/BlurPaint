import sys
import torch
import os
import glob

from PIL import Image
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf
from utils import instantiate_from_config
from calc_metrics import metrics_calc

from torchvision import transforms
import albumentations as A
import argparse


def tensor_to_image():
    return transforms.ToPILImage()

def image_to_tensor():
    return transforms.ToTensor()

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask_L = np.array(Image.open(mask).convert("L")).astype(np.uint8)
    transform = A.Resize(height=256, width=256)
    mask_L = transform(image=mask_L)['image']
    mask_L = mask_L.astype(np.float32)/255.0
    mask_L = mask_L[None, None]
    mask_L[mask_L < 0.5] = 0
    mask_L[mask_L >= 0.5] = 1
    mask_L = torch.from_numpy(mask_L)

    mask_RGB = np.array(Image.open(mask).convert("RGB"))
    mask_RGB = mask_RGB.astype(np.float32)/255.0
    mask_RGB = mask_RGB[None].transpose(0, 3, 1, 2)
    mask_RGB = torch.from_numpy(mask_RGB)

    masked_image = (1-mask_L)*image

    batch = {"image": image, "mask": mask_L, "masked_image": masked_image,
             "mask_RGB": mask_RGB}

    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch

def test():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='./', help='The checkpoint path')
    parser.add_argument("--input_path", type=str, default='./',help='The test input image path')
    parser.add_argument("--mask_path", type=str, default='./',help='The test input mask path')
    parser.add_argument("--output_path", type=str, default='./',help='The output image path')

    opts = parser.parse_args()

    config = OmegaConf.load(os.path.join(sys.path[0], "confs/denoise-network.yaml"))

    argparse_dict = vars(opts)
    merged_config = OmegaConf.merge(config, OmegaConf.create(argparse_dict))

    test_img_path = merged_config.input_path
    test_mask_path = merged_config.mask_path

    test_data = sorted(glob.glob(os.path.join(sys.path[0], test_img_path, "*.*")))
    masks = sorted(glob.glob(os.path.join(sys.path[0], test_mask_path, "*.*")))
    print(f"Finding {len(test_data)} images and {len(masks)} masks")

    model = instantiate_from_config(config.model)

    output_path = merged_config.output_path

    model.load_state_dict(torch.load(merged_config.ckpt_path)["state_dict"],
                          strict=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    os.makedirs(output_path, exist_ok=True)

    print("Strat sampling.........")

    with torch.no_grad():
        for image, mask in tqdm(zip(test_data, masks)):
            outpath = os.path.join(sys.path[0], output_path, os.path.split(image)[1])
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask

            x0 = batch["masked_image"]
            shape = batch["masked_image"].shape[1:]
            cond = torch.cat((batch["masked_image"], batch["mask"]), dim=1)

            mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                               min=0.0, max=1.0)

            masked_image = torch.clamp((batch["masked_image"] + 1.0) / 2.0,
                                       min=0.0, max=1.0)


            predicted_image = model.p_sample_loop(
                            step=20,
                            cond=cond,
                            batch_size=cond.shape[0],
                            shape=shape,
                            mask=mask,
                            x0=x0,
                            )

            predicted_image = torch.clamp((predicted_image+1.0)/2.0, min=0.0, max=1.0)

            inpainted = (1-mask)*masked_image+mask*predicted_image

            output = (inpainted[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = Image.fromarray(output)
            output.save(outpath)

    metrics_calc(os.path.join(sys.path[0], test_img_path), output_path)

if __name__ == "__main__":
    test()