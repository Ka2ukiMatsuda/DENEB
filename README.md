# 🤖 [ACCV2024] DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning
[![arXiv](https://img.shields.io/badge/arXiv-2409.19255-B31B1B)](https://arxiv.org/abs/2409.19255)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow)](https://huggingface.co/datasets/Ka2ukiMatsuda/Nebula)


*We address the challenge of developing automatic evaluation metrics for image captioning, with a particular focus on the robustness against hallucinations. Existing metrics often inadequately handle hallucinations, primarily due to their limited ability to compare candidate captions with multifaceted reference captions. To address this shortcoming, we propose DENEB, a novel supervised automatic evaluation metric specifically robust against hallucinations. DENEB incorporates the Sim-Vec Transformer, a mechanism that processes multiple references simultaneously, thereby efficiently capturing the similarity among an image, a candidate caption, and reference captions. Furthermore, to train DENEB, we construct the Nebula dataset, a diverse and balanced dataset comprising 32,978 images, paired with human judgments provided by 805 annotators. We achieved state-of-the-art performance on FOIL, Composite, Flickr8K-Expert, Flickr8K-CF, Nebula, and PASCAL-50S, thereby demonstrating its effectiveness and robustness against hallucinations.*

![eye-catch](https://i.imgur.com/DWKCax6.png)

## Instructions

We assume the following environment for our experiments:

- Python 3.10.0 (pyenv is strongly recommended)
- [Poetry](https://github.com/python-poetry/poetry) for dependency management (refer to Poetry documentation)
- PyTorch version 2.1.0 with CUDA 11.8 support
- PyTorch Lightning for model training facilitation

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

### Dataset

  - The Nebula dataset can be downloaded from Hugging Face: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow)](https://huggingface.co/datasets/Ka2ukiMatsuda/Nebula)


### Train

```bash
sh train.sh
```

### Evaluation

PAC-S checkpoints are required to assess PAC-S. 


Download the checkpoints according to the instructions on the [authors' github](https://github.com/aimagelab/pacscore) and place them in the specified locations.

```bash
sh validate.sh
```

## Citation

```bash
@inproceedings{matsuda2024deneb,
  title={DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning},
  author={Kazuki Matsuda and Yuiga Wada and Komei Sugiura},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2024},
  pages={3570--3586}
}
```
