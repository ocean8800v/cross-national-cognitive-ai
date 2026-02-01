# manuscript-pipeline
![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)

This repository contains code and scripts supporting the manuscript *“Low-Burden AI Approach for Cross-National Early Identification of Cognitive Impairment Using Real-World Questionnaire Response Behaviours”*.

It includes Python notebooks, R scripts, and utility functions. Python analyses were conducted in Google Colab (Intel Xeon CPU, 12 cores, 2.20 GHz; NVIDIA A100 GPU, 40 GB, CUDA 12.4).


## Software environment

- **Python**
  - General analyses: version **3.12.11**
  - Fine-tuning experiments: version **3.11.13**
  - Dependencies are listed in [`requirements_analysis.txt`](requirements_analysis.txt)

- **R**
  - Version **4.2.1**
  - RStudio **2025.5.0.496**
  - Key packages:  
    - `ggplot2 3.5.2`, `ComplexHeatmap 2.14.0`, `patchwork 1.1.3`  
    - `shapviz 0.9.2`, `gridExtra 2.3`, `cowplot 1.1.3`, `circlize 0.4.16`  
    - `lme4 1.1-37`, `performance 0.14.0`


## Main analysis

- [`1_manuscript_analysis.ipynb`](1_manuscript_analysis.ipynb) – main pipeline  
- Shared functions in [`manuscript_utils`](manuscript_utils/)  


## Installation

Each notebook includes installation instructions. For example, in `1_manuscript_analysis.ipynb`:
```python
!git clone https://github.com/ocean8800v/cross-national-cognitive-ai.git
!pip install -r cross-national-cognitive-ai/requirements_analysis.txt
```

Typical install time: ~5–10 minutes.

## Model weights

- [default_tabpfn-v2-classifier_weights.ckpt](https://drive.google.com/file/d/1L_TbN2Du7ZOjNm1VXz_rxOELgFHWey30/view?usp=drive_link) – [Prior-Labs/TabPFN](https://github.com/PriorLabs/TabPFN) (accessed 9 July 2025)  
- [fine-tuned_tabpfn-v2-classifier_weights.ckpt](https://drive.google.com/file/d/1wnFKigC27-mqgNPAI23Rq94RGuG5o6fx/view?usp=drive_link) – fine-tuned weights generated in this study    


## Fine-tuning experiments

[`2_finetune_experiment.ipynb`](2_finetune_experiment.ipynb) – fine-tuning pipeline

The fine-tuning experiment code is located in [`finetune_experiment`](finetune_experiment/).  

This code is adapted from [LennartPurucker/finetune_tabpfn_v2](https://github.com/LennartPurucker/finetune_tabpfn_v2) (accessed 25 August 2025), originally released under the BSD 3-Clause Licence.  

All other original code in this repository is licensed under the MIT Licence.

## Data availability

Data used in this study are available upon registration from the Gateway to Global Aging Data (https://g2aging.org/).

## Low-Quality Response (LQR) indicators

The methods for the seven Low-Quality Response (LQR) indicators described in the manuscript were originally developed by:

> Schneider S, Lee P-J, Hernandez R, et al.  
> *Cognitive Functioning and the Quality of Survey Responses: An Individual Participant Data Meta-Analysis of 10 Epidemiological Studies of Aging*.  
> The Journals of Gerontology: Series B, 2024.  
> https://doi.org/10.1093/geronb/gbae030
