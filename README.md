# [ACCV2024] DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning

[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90Project%20Page-DENEB-23E8CE)](https://deneb-project-page-nc03k.kinsta.page/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Nebula-yellow)](https://huggingface.co/datasets/Ka2ukiMatsuda/Nebula)
[![arXiv](https://img.shields.io/badge/arXiv-2409.19255-B31B1B)](https://arxiv.org/abs/2409.19255)
[![Available in CaptionEvalKit](https://img.shields.io/badge/available%20in-CaptionEvalKit-blue)](https://github.com/YuigaWada/CaptionEvalKit-for-VLMs)

<p align="center">
  <img src="https://i.imgur.com/DWKCax6.png" alt="Example Image" width="500"/>
</p>

## 🎉🎉 News

- 2026-06-13: The Nebula benchmark is supported in [CaptionEvalKit-for-VLMs](https://github.com/YuigaWada/CaptionEvalKit-for-VLMs), a reproducible caption-evaluation toolkit. You can evaluate metrics on Nebula with a single command.

## 🔁 Evaluate on Nebula via CaptionEvalKit

The Nebula benchmark is integrated in [CaptionEvalKit-for-VLMs](https://github.com/YuigaWada/CaptionEvalKit-for-VLMs). CaptionEvalKit loads the dataset from Hugging Face `Ka2ukiMatsuda/Nebula` and computes Kendall correlations against human judgments.

```bash
pip install capevalkit
capevalkit all_reproduce --benchmarks nebula
```

This runs BLEU, CIDEr, CLIPScore, PAC-S, Polos, and other metrics that ship Nebula expected values. CaptionEvalKit runs each metric in an isolated uv environment, so benchmark evaluation does not require manually recreating each metric's original environment.

## 🗂️ Nebula Dataset [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Nebula-yellow)](https://huggingface.co/datasets/Ka2ukiMatsuda/Nebula)

We introduce **Nebula**, a diverse and balanced dataset for training and evaluating hallucination-robust caption metrics.

- 🖼 32,978 images
- 📝 Multifaceted reference captions and candidate captions
- 🧑‍⚖️ Human quality judgments from 805 annotators

### Load from Hugging Face

```python
from datasets import load_dataset

nebula = load_dataset("Ka2ukiMatsuda/Nebula")
print(nebula)
# DatasetDict({
#   train: 26,382 samples
#   valid: 3,298 samples
#   test:  3,298 samples
# })
```

Each sample contains `file_name`, `image`, `refs`, `mt`, and `human_score`.

## 🧪 DENEB (Training & Evaluation)

### Requirements

- Python 3.10.0 (pyenv is strongly recommended)
- [Poetry](https://github.com/python-poetry/poetry) for dependency management
- PyTorch 2.1.0 with CUDA 11.8 support
- PyTorch Lightning for model training

### Installation

```bash
git clone --recursive XXXXX
cd DENEB
```

```bash
pyenv virtualenv 3.10.0 deneb
pyenv local deneb
sh install.sh # cuda=11.8
```

### Training

```bash
sh train.sh
```

### Evaluation

PAC-S checkpoints are required to assess PAC-S.

Download the checkpoints according to the instructions on the [authors' github](https://github.com/aimagelab/pacscore) and place them in the specified locations.

```bash
sh validate.sh
```

## 📄 Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo:

```bibtex
@inproceedings{matsuda2024deneb,
  title={DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning},
  author={Kazuki Matsuda and Yuiga Wada and Komei Sugiura},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2024},
  pages={3570--3586}
}
```
