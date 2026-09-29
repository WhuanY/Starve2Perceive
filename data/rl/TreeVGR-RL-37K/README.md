---
language:
- en
license: apache-2.0
task_categories:
- image-text-to-text
tags:
- visual-grounding
- vqa
- reasoning
- benchmark
library_name: datasets
---

# TreeBench Dataset Card

This repository contains **TreeBench** (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark designed for evaluating "thinking with images" capabilities with *traceable visual evidence*.

The dataset was introduced in the paper: [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](https://arxiv.org/abs/2507.07999).

TreeBench is built on three core principles:
1.  **Focused visual perception**: of subtle targets in complex scenes.
2.  **Traceable evidence**: via bounding box evaluation.
3.  **Second-order reasoning**: to test object interactions and spatial hierarchies beyond simple object localization.

Prioritizing images with dense objects, TreeBench initially sampled 1K high-quality images from SA-1B. Eight LMM experts manually annotated questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, with which even the most advanced models struggle.

![TreeBench Overview](https://github.com/Haochen-Wang409/TreeVGR/raw/main/assets/treebench.png)

For further details, the full codebase, and related models (like TreeVGR), please refer to the [official GitHub repository](https://github.com/Haochen-Wang409/TreeVGR).

## Usage

This repository provides a simple local inference demo of TreeVGR on TreeBench. To get started:

First, clone the repository:
```bash
git clone https://github.com/Haochen-Wang409/TreeVGR
cd TreeVGR
```
Then, install the required packages:
```bash
pip3 install -r requirements.txt
pip3 install flash-attn --no-build-isolation -v
```
Finally, run the inference script:
```bash
python3 inference_treebench.py
```
This should produce output similar to:
```
Perception/Attributes 18/29=62.07
Perception/Material 7/13=53.85
Perception/Physical State 19/23=82.61
Perception/Object Retrieval 10/16=62.5
Perception/OCR 42/68=61.76
Reasoning/Perspective Transform 19/85=22.35
Reasoning/Ordering 20/57=35.09
Reasoning/Contact and Occlusion 25/41=60.98
Reasoning/Spatial Containment 20/29=68.97
Reasoning/Comparison 20/44=45.45
==> Overall 200/405=49.38
==> Mean IoU: 43.3
```

## Citation

If you find TreeBench useful for your research and applications, please cite the following paper:

```bibtex
@article{wang2025traceable,
  title={Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology},
  author={Haochen Wang and Xiangtai Li and Zilong Huang and Anran Wang and Jiacong Wang and Tao Zhang and Jiani Zheng and Sule Bai and Zijian Kang and Jiashi Feng and Zhuochen Wang and Zhaoxiang Zhang},
  journal={arXiv preprint arXiv:2507.07999},
  year={2025}
}
```