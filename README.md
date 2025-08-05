<div align="center">
<h1>
LightMotion: A Light and Tuning-free Method for Simulating Camera Motion in Video Generation [Official Code of PyTorch]
</h1>

<div>
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Quanjian Song<sup>1</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?user=UpqNGLYAAAAJ&hl=zh-CN&oi=ao' target='_blank' style='text-decoration:none'>Zhihang Lin<sup>1,2</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=P9ctuRUAAAAJ&view_op=list_works&sortby=pubdate' target='_blank' style='text-decoration:none'>Zhanpeng Zeng<sup>1</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=6b9pVLIAAAAJ' target='_blank' style='text-decoration:none'>Ziyue Zhang<sup>1</sup></a>, &ensp;
    <a href='https://mac.xmu.edu.cn/ljcao/' target='_blank' style='text-decoration:none'>Liujuan Cao<sup>1,†</sup></a>, &ensp;
    <a href='https://mac.xmu.edu.cn/rrji/' target='_blank' style='text-decoration:none'>Rongrong Ji<sup>1</sup></a>
</div>

<div>
    <sup>1</sup> Key Laboratory of Multimedia Trusted Perception and Efficient Computing, <br> Ministry of Education of China, Xiamen University, China.
    <br>
    <sup>2</sup> Shanghai Innovation Institute.  &ensp;
    <sup>†</sup> Corresponding Author.
    
</div>

<sub></sub>

<p align="center">
    <span>
        <a href="https://arxiv.org/abs/2503.06508" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202503.06508-LightMotion-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span> 
        <a href='https://github.com/QuanjianSong/LightMotion' target="_blank">
        <img src='https://img.shields.io/badge/Project_Page-LightMotion-green' alt='Project Page'></a>  &emsp;  &emsp;
    </span>
    <span> 
        <a href='https://huggingface.co/papers/2503.06508' target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-LightMotion-yellow' alt='Project Page'></a> &emsp;  &emsp;
    </span>
</p>
</div><div align="center">
<h1>
StyDeco: Unsupervised Style Transfer with Distilling Priors and Semantic Decoupling [Official Code of PyTorch]
</h1>

<div>
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Yuanlin Yang<sup>*</sup></a>, &ensp;
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Quanjian Song<sup>*</sup></a>, &ensp;
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Zhexian Gao</a>, &ensp;
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Ge Wang</a>, &ensp;
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Shanshan Li</a>, &ensp;
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Xiaoyan Zhang</a>
</div>

<div>
    <sup>*</sup> Equal contribution.
</div>

<sub></sub>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2508.01215" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202508.01215-StyDeco-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span> 
        <a href='https://github.com/QuanjianSong/LightMotion' target="_blank">
        <img src='https://img.shields.io/badge/Project_Page-StyDeco-green' alt='Project Page'></a>  &emsp;  &emsp;
    </span>
    <span> 
        <a href='https://huggingface.co/papers/2508.01215' target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-StyDeco-yellow' alt='Hugging Face'></a> &emsp;  &emsp;
    </span>
</p>
</div>



## 🎉 News
<pre>
• <strong>2025.08</strong>: 🔥 The official code of StyDeco has been released.
• <strong>2025.08</strong>: 🔥 The paper of StyDeco has been submitted to <a href="https://arxiv.org/pdf/2508.01215">arXiv</a>.
</pre>


## 🎬 Overview
In this paper, we propose StyDeco, an unsupervised framework that resolves this limitation by learning text representations specifically tailored for the style transfer task. Our framework first employs Prior-Guided Data Distillation (PGD), a strategy designed to distill stylistic knowledge without human supervision. It leverages a powerful frozen generative model to automatically synthesize pseudopaired data. Subsequently, we introduce Contrastive Semantic Decoupling (CSD), a task-specific objective that adapts a text encoder using domain-specific weights. CSD performs a twoclass clustering in the semantic space, encouraging source and target representations to form distinct clusters. The overall framework is illustrated as follows:
![Overall Framework](imgs/overall_framework.png)

## 🔧 Environment
```
git clone https://github.com/QuanjianSong/StyDeco.git
# Installation with the requirement.txt
conda create -n StyDeco python=3.10
conda activate StyDeco
pip install -r requirements.txt
# Or installation with environment.yaml
conda env create -f environment.yaml
```

## 🔥 Train
```
bash train_StyDeco.sh
```

## 🚀 Inference
```
bash start_StyDeco.sh
```

## 🎓 Bibtex
🤗 If you find this code helpful for your research, please cite:
```
XXX
```