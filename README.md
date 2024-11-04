# LIIFusion: Learning Implicit Image Function Using Image Prior

Inspired by [LIIF](https://github.com/yinboc/liif), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [IDM](https://github.com/Ree1s/IDM), we fuse INR and latent diffusion for arbitrary scale, efficient and realistic image SR. Our model needs two stage training. At the first stage, our model learns how to encode a prior from a HR image. By injecting the prior to a latent representation of a LR image, LIIF can upsample the LR image more correctly. At the second stage, the prior encoding module is replaced to diffusion module. We expect that by sampling the prior from the diffusion module, we can generate plausible details for SR images. Besides, as we conduct the diffusion prcoess on a prior, we can reduce computational cost and the number of denoising step than IDM. To alleviate over-smoothing problem, we give GAN loss to our model. Quantitative and qualitative results are as follows. 

## Quantitative Results

* #### DIV2K
  <img width="898" alt="스크린샷 2024-11-03 오전 12 01 41" src="https://github.com/user-attachments/assets/70c464cd-5825-4a28-b5f0-b19c6b1530d7">

* #### Benchmarks
  <img width="741" alt="스크린샷 2024-11-02 오후 11 56 55" src="https://github.com/user-attachments/assets/0258b98a-90cf-4a78-b194-58f28fc044c6">

* #### CelebA-HQ(1-100 images)
  <img width="756" alt="스크린샷 2024-11-02 오후 11 29 49" src="https://github.com/user-attachments/assets/6f9963c0-7063-4bd8-a743-35075e676e6f">

* #### Inference time per image
  <img width="364" alt="스크린샷 2024-11-02 오후 11 57 38" src="https://github.com/user-attachments/assets/1eb123cd-d51c-42a9-ae56-806e6cc94749">


### Qualitative Results

* #### Stage 2 B100
  <img width="570" alt="스크린샷 2024-11-02 오후 11 31 34" src="https://github.com/user-attachments/assets/ca23f166-b53d-47eb-8c19-667245b85cbe">

* #### Stage 2 Urban100
  <img width="527" alt="스크린샷 2024-11-02 오후 11 30 08" src="https://github.com/user-attachments/assets/fa6215d8-d833-4044-b13a-1bdab01ae8a6">

* #### Stage 2 CelebA-HQ
  <img width="673" alt="스크린샷 2024-11-02 오후 11 30 35" src="https://github.com/user-attachments/assets/a91a73c2-a251-4bf6-86a9-767b8879871e">



### Data
We use **DIV2K**, **FFHQ**, **CelebA-HQ** and **Benchmark**. The **Benchmark** includes **Set5**, **Set14**, **Urban100**, and **B100**.
`mkdir load` for putting the dataset folders.

### How to train 
* Stage1
  <pre><code>python train.py --config [path-to-config]</pre></code>
* Stage2
  <pre><code>python train.py --config [path-to-config]</pre></code>
### How to test
* DIV2K
  <pre><code>bash scripts/test-div2k.sh [MODEL_PATH] [GPU]</pre></code>
* Benchmark
  <pre><code>bash scripts/test-benchmark.sh [MODEL_PATH] [GPU] </pre></code>
* CelebA-HA(1-100 images)
  <pre><code>bash scripts/test-celebahq.sh [MODEL_PATH] [GPU] </pre></code>
### How to upsample an image
  <pre><code> python demo.py --input [IMAGE_PATH] --model [MODEL_PATH] --scale [SCALE_NUM]</pre></code>

### Acknowledgement
This code is based on these [LIIF](https://github.com/yinboc/liif), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [SwinIR](https://github.com/novwaul/SwinIR) repos.
