---
language:
- en
license: apache-2.0
task_categories:
- image-text-to-text
tags:
- vqa
- visual-grounding
- reasoning
- benchmark
- multimodal
library_name: datasets
---

# TreeBench: Traceable Evidence Enhanced Visual Grounded Reasoning Benchmark

This repository contains **TreeBench**, a diagnostic benchmark dataset proposed in the paper [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](https://arxiv.org/abs/2507.07999).

TreeBench is designed to holistically evaluate "thinking with images" capabilities by dynamically referencing visual regions. It is built on three core principles:
1.  **Focused visual perception** of subtle targets in complex scenes.
2.  **Traceable evidence** via bounding box evaluation.
3.  **Second-order reasoning** to test object interactions and spatial hierarchies beyond simple object localization.

The benchmark consists of 405 challenging visual question-answering pairs, meticulously annotated by eight LMM experts. Images are initially sampled from SA-1B, prioritizing those with dense objects. Even the most advanced models struggle with TreeBench, highlighting the current limitations in visual grounded reasoning.

For detailed usage, please refer to our GitHub repository: [https://github.com/Haochen-Wang409/TreeVGR](https://github.com/Haochen-Wang409/TreeVGR)

For the images used in TreeBench, please refer to: [https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data)

## Sample Usage

This dataset is primarily used for evaluating Visual Grounded Reasoning models. For a simple local inference demo of the associated TreeVGR model on TreeBench, clone the GitHub repository and run the `inference_treebench.py` script.

```bash
git clone https://github.com/Haochen-Wang409/TreeVGR
cd TreeVGR
python3 inference_treebench.py
```

## Related Resources

This dataset is part of a larger project with associated models and training datasets available on the Hugging Face Hub:

-   **TreeVGR Models**:
    -   [TreeVGR-7B](https://huggingface.co/HaochenWang/TreeVGR-7B)
    -   [TreeVGR-7B-CI](https://huggingface.co/HaochenWang/TreeVGR-7B-CI)
-   **Training Datasets**:
    -   [TreeVGR-RL-37K](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K)
    -   [TreeVGR-SFT-35K](https://huggingface.co/datasets/HaochenWang/TreeVGR-SFT-35K)

## Citation

If you find TreeBench useful for your research and applications, please cite the associated paper:

```bibtex
@article{wang2025traceable,
  title={Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology},
  author={Haochen Wang and Xiangtai Li and Zilong Huang and Anran Wang and Jiacong Wang and Tao Zhang and Jiani Zheng and Sule Bai and Zijian Kang and Jiashi Feng and Zhuochen Wang and Zhaoxiang Zhang},
  journal={arXiv preprint arXiv:2507.07999},
  year={2025}
}
```