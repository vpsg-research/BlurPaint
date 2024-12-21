
# BlurPaint: Image Inpainting using Blurring Diffusion Model


## Running the Experiments
The code has been tested on Python 3.8.10 and PyTorch 2.0.0. 
Please refer to `environment.yml` for a list of environments that can be used to run the code.

## Usage

The models and datasets are placed in the `data/` folder as follows:
```bash
<data> # a folder contains of all data
├── celeba-hq 
│   ├── train_img # all train images
│   │   └── *.jpg
├── mask 
│   ├── mask_train #all train mask images
│   │   └── *.png
```
For mask, 1 indicates the masked regions, and 0 indicates the unmasked
regions. All images are 256X256.

## Run Train:

```
  $ cd BlurPaint
  $ python main.py --input_path [train_img_path] \
                   --mask_path [train_mask_path] \
                   --device [gpu_ids]
```
Example:

if you want to train on multiple gpus:

python train.py --input_path image_path --mask_path mask_path --device 0,1,2

if you want to train on one gpu:

python train.py --input_path image_path --mask_path mask_path --device 0


## Run Test:
```
  $ cd BlurPaint
  $ python test.py --ckpt_path [save_path] \
                   --input_path [test_img_path] \
                   --mask_path [test_mask_path] \
                   --output_path [output_path]
```
## Citation
Please cite our paper if the code is used in your research:
```
@ARTICLE{Chen2025BlurPaint,
  author={Chen, Linxu and Guo, Zhiqing and Wang, Liejun},
  booktitle={ICASSP}, 
  title={BlurPaint: Image Inpainting using Blurring Diffusion Models}, 
  year={2025}}
```
