# [CVPR2024] Adversarial Score Distillation: When score distillation meets GAN
[Arxiv](https://arxiv.org/abs/2312.00739) | [Paper](https://2y7c3.github.io/pdfs/asd.pdf) | [Project page](https://2y7c3.github.io/ASD/asd.html)
---
### 🎉 We released the training codes using 3DGS. 
 💡 Tips: Camera parameters are crucial for the final results of 3DGS.

## Overview (see Project page for more examples)
Generated 3D NeRFs
<div align=center>
<img src="https://2y7c3.github.io/ASD/imgs/ex7.gif" width="45%"></img><img src="https://2y7c3.github.io/ASD/imgs/ex10.gif" width="45%"></img>
</div>
Generated 3D Gaussians
<div align=center>
<img src="https://2y7c3.github.io/ASD/imgs/gs5.gif" width="25%"></img><img src="https://2y7c3.github.io/ASD/imgs/gs1.gif" width="25%"></img>
</div>

## Installation
Install ASD requirements, [Differential Gaussian Rasterization](https://github.com/YixunLiang/diff-gaussian-rasterization) and [simple-knn](https://github.com/YixunLiang/simple-knn)
```sh
git clone https://github.com/2y7c3/ASD
cd ASD

### for 3D Gaussian Splatting
git clone --recursive https://github.com/YixunLiang/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

git clone https://github.com/YixunLiang/simple-knn.git
pip install ./simple-knn

pip install -r requirements.txt
```

### Optional
```sh
pip install ninja
```
Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

#### For 3D Gaussian Splatting

Install [Point-E](https://github.com/openai/point-e)

Install [Shape-E](https://github.com/openai/shap-e)

Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models) by Cap3D, and put it in `./load`

## Quick start

```sh
# NeRF Training
python launch.py --config configs/test_nerf.yaml --train --gpu 0 system.prompt_processor.prompt="A delicious hamburger"

3D gaussian Training (experimental implementation)
python launch.py --config configs/test_gs.yaml --train --gpu 0 system.prompt_processor.prompt="A delicious hamburger"

# Tuning
# you might want to resume training from the certain checkpoint
python launch.py --config configs/test_tune_{nerf or gs}.yaml --train --gpu 0 system.prompt_processor.prompt="A delicious hamburger" resume="path/to/ckpt"

# Testing 
# you can change camera parameters on here
python launch.py --config configs/test_tune_{nerf or gs}.yaml --test --gpu 0 system.prompt_processor.prompt="A delicious hamburger" resume="path/to/ckpt"
```

## Todo
- [x] Release the training codes for NeRF
- [x] Release the training codes for 3DGS
- [ ] Release the training codes for 2D images and image editing

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wei_2024_CVPR,
    author    = {Wei, Min and Zhou, Jingkai and Sun, Junyao and Zhang, Xuesong},
    title     = {Adversarial Score Distillation: When score distillation meets GAN},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8131-8141}
}
```

## Acknowledgements
This code is built on many research works and open-source projects:
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
- [ThreeStudio](https://github.com/threestudio-project/threestudio)
- [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer/)

Thanks for their excellent works.
