# LIIFusion
### Efficient and Realistic Image Super-Resolution for Arbitrary Scale

Inspired by [LIIF](https://github.com/yinboc/liif), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [IDM](https://github.com/Ree1s/IDM), we fuse INR and latent diffusion for arbitrary scale, efficient and realistic image SR. Our model needs two stage training. At the first stage, our model learns how to encode a prior from a HR image. By injecting the prior to a latent representation of a LR image, LIIF can upsample the LR image more correctly. At the second stage, the prior encoding module is replaced to diffusion module. We expect that by sampling the prior from the diffusion module, we can generate plausible details for SR images. Besides, as we conduct the diffusion prcoess on a prior, we can reduce computational cost and the number of denoising step than IDM. To alleviate over-smoothing problem, we give GAN loss to our model. Quantiative and qualitative results are as follows. 

* #### Comparison with fixed scale diffusion models
<img width="954" alt="stage2" src=https://github.com/novwaul/LIIFusion/assets/53179332/2b5df0d0-a0b4-4c47-a6e0-069b3f0e043a>

* #### Comparison with IDM
<img width="954" alt="stage2" src=https://github.com/novwaul/LIIFusion/assets/53179332/9160df33-e19c-4f79-b173-947c8ba44237>



* #### Stage 2 DIV2K
<img width="954" alt="stage2" src=https://github.com/novwaul/LIIFusion/assets/53179332/1629de81-ca1f-4cf3-980d-3e57f020c72d>


* #### Stage 2 B100
<img width="954" alt="srage2_2" src=https://github.com/novwaul/LIIFusion/assets/53179332/389b8e3e-c4cd-4f93-b83c-531f9ffd8b39>


### Data
We use **DIV2K** and **Benchmark**. The **Benchmark** includes **Set5**, **Set14**, **Urban100**, and **B100**.
`mkdir load` for putting the dataset folders.

- **DIV2K**: `mkdir` and `cd` into `load/div2k`. Download and `unzip` the [Train_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [Valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip), [Valid_LR_X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip), [Valid_LR_X3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip), and [Valid_LR_X4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip) (provided by [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)). `mv` `X4/`, `X3/`, and `X2/` folders of Valid_LR to a single `DIV2K_valid_LR_bicubic` folder.
- **Benchmark**: `cd` into `load/`. Download and `tar -xf` the [Benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (provided by [this repo](https://github.com/thstkdgus35/EDSR-PyTorch)).

### How to train 
* Stage1
<pre><code>python train.py --config /home/kaist2/Desktop/LIIFusion/configs/train-two-stage/train_stage1.yaml</pre></code>
* Stage2
<pre><code>python train.py --config /home/kaist2/Desktop/LIIFusion/configs/train-two-stage/train_stage2.yaml</pre></code>
### How to test
* DIV2K
<pre><code>bash scripts/test-div2k.sh [MODEL_PATH] [GPU]</pre></code>
* Benchmark
<pre><code>bash scripts/test-benchmark.sh [MODEL_PATH] [GPU] </pre></code>
### How to upsample an image
<pre><code> python demo.py --input [IMAGE_PATH] --model [MODEL_PATH] --scale [SCALE_NUM]</pre></code>

### Acknowledgement
This code is based on these [LIIF](https://github.com/yinboc/liif), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [SwinIR](https://github.com/novwaul/SwinIR) repos.
