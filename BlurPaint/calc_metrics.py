
import numpy as np
import torch
import glob
import os
import cv2

from PIL import Image
from lpips import LPIPS
from tqdm import tqdm

from pytorch_fid import fid_score
from skimage.metrics import structural_similarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_fn = LPIPS(net='alex').to(device).eval()

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def open_img_as_np(path):
    image = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    # image = image.astype(np.float32) / 255.0
    # image = image.transpose(2, 0, 1)  # h,w,c  -> c,h,w
    # # image = np.expand_dims(image, axis=0)
    #
    # image = image * 2.0 - 1.0

    return image

def read_ssim_im(fpath):
    img = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024))
    return img

def open_img_as_torch(path):
    image = np.array(Image.open(path).convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # h,w,c  -> c,h,w
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0

    return image

def metrics_calc(gt_img_path, rec_img_path):
    total_ssim_num = []
    total_lpips = 0.0
    total_psnr = 0.0

    input = sorted(list(glob.glob(os.path.join(gt_img_path, '*.*'))))
    rec = sorted(list(glob.glob(os.path.join(rec_img_path, '*.*'))))

    print("total rec img nums is ", len(rec))
    print("total gt img nums is ", len(input))
    img_nums = len(input)

    for i in tqdm(range(img_nums)):
        input_t = read_ssim_im(input[i])
        rec_img = read_ssim_im(rec[i])
        total_ssim_num.append(structural_similarity(input_t, rec_img))

    total_ssim = np.array(total_ssim_num).mean()

    total_fid = fid_score.calculate_fid_given_paths([gt_img_path, rec_img_path],
                                                device=device, batch_size=20, dims=2048)

    # calc_psnr and calc_lpips
    for i in range(img_nums):
        input_t = open_img_as_np(input[i])
        rec_img = open_img_as_np(rec[i])

        temp_psnr = calculate_psnr(input_t, rec_img)
        # temp_ssim = calculate_ssim(input_t, rec_img)
        # total_ssim += temp_ssim
        total_psnr += temp_psnr

        input_t = open_img_as_torch(input[i])
        rec_img = open_img_as_torch(rec[i])

        temp_lp_score = lpips_fn(input_t.to(device), rec_img.to(device))
        total_lpips += temp_lp_score.item()

    total_lpips /= img_nums
    total_psnr /= img_nums

    print("↑--ssim={:.4f},  ↓--lpips={:.3f},  ↑--psnr={:.2f},  ↓--FID={:.3f}"
          .format(total_ssim, total_lpips, total_psnr, total_fid))

    # return total_ssim, total_lpips, total_psnr, total_fid