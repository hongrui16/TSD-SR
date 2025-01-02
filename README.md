<div align="center">


<h1>TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution</h1>

<div>
    <a href='https://github.com/Microtreei' target='_blank'>Linwei Dong<sup>1,*</sup></a>&emsp;
    <a href='https://fqnchina.github.io/' target='_blank'>Qingnan Fan<sup>2,*</sup></a>&emsp;
    <a href='https://github.com/Sun-Made-By-Yi' target='_blank'>Yihong Guo<sup>1</sup></a>&emsp;
    <a href='https://github.com/Wzh10032' target='_blank'>Zhonghao Wang<sup>3</sup></a>&emsp;
    <a href='https://qzhang-cv.github.io/' target='_blank'>Qi Zhang<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=Pcsml4oAAAAJ' target='_blank'>Jinwei Chen<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=pnVwaGsAAAAJ&hl=en' target='_blank'>Yawei Luo<sup>1,†</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=kj5HiGgAAAAJ&hl=en' target='_blank'>Changqing Zou<sup>1,4 </sup></a>
</div>
<div>
    <sup>1</sup>Zhejiang University, <sup>2</sup>Vivo Mobile Communication Co. Ltd, <sup>3</sup>University of Chinese Academy of Sciences,  <sup>4</sup>Zhejiang Lab 
</div>

[[paper]](https://arxiv.org/abs/2411.18263)

---

</div> 

## 🔥 <a name="news"></a>News
- **[2025.01]** Release the TSD-SR, including the inference codes and pretrained models.
- **[2024.12]** This repo is created.

:hugs: If TSD-SR is helpful to your projects, please help star this repo. Thanks! :hugs:

## 🎬 <a name="overview"></a>Overview
![overview](assets/pipeline.png)

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/Microtreei/TSD-SR.git
cd TSD-SR

# create an environment 
conda create -n tsdsr python=3.9
conda activate tsdsr
pip install -r requirements.txt
```

## 🚀 <a name="start"></a>Quick Start
#### Step 1: Download the pretrained models
- Download the pretrained SD3 models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main).
- Download the TSD-SR lora weights and prompt embeddings from [GoogleDrive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI?usp=drive_link).

You can put the models weights into `checkpoint/tsdsr`.
You can put the prompt embbedings into `dataset/default`.

#### Step 2: Prepare testing data
You can put the testing images in the `imgs`.

#### Step 3: Running testing command
```
python test/test_tsdsr.py \
--pretrained_model_name_or_path /path/to/your/sd3 \
-i imgs \
-o outputs \
--lora_dir checkpoint/tsdsr \
--embedding_dir dataset/default/ 
```

#### Step 4: Running testing metrics command
```
python test/test_metrics.py \
--inp_imgs outputs \
--gt_imgs /path/to/your/gt/images \
--log logs/metrics
```

## <a name="results"></a>🔎 Results
<details>
    <summary> Quantitative comparison with the state-of-the-art <b>one-step</b> methods across both synthetic and real-world benchmarks (click to expand). </summary>
    <p align="center">
    <img width="900" src="assets/one_step.png">
    </p>
</details>

<details>
    <summary> Quantitative comparison with the state-of-the-art <b>multi-step</b> methods across both synthetic and real-world benchmarks (click to expand). </summary>
    <p align="center">
    <img width="900" src="assets/multi_step.png">
    </p>
</details>

<details>
    <summary> Visual comparisons of different <b>Diffusion-based</b> Real-ISR methods. </summary>
    <p align="center">
    <img width="900" src="assets/visualization1.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization2.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization3.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization4.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization5.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization6.png">
    </p>
</details>

<details>
    <summary> Visual comparisons of different <b>GAN-based</b> Real-ISR methods. </summary>
    <p align="center">
    <img width="900" src="assets/visualization-gan1.png">
    </p>
    <p align="center">
    <img width="900" src="assets/visualization-gan2.png">
    </p>
</details>



## 🎫 <a name="license"></a>License
This project is released under the [Apache 2.0 license](LICENSE).

## <a name="citation"></a>🎓 Citation


```
@article{dong2024tsd,
  title={TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution},
  author={Dong, Linwei and Fan, Qingnan and Guo, Yihong and Wang, Zhonghao and Zhang, Qi and Chen, Jinwei and Luo, Yawei and Zou, Changqing},
  journal={arXiv preprint arXiv:2411.18263},
  year={2024}
}
```
