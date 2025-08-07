# G-softmax-DDG [![DOI](https://zenodo.org/badge/949723818.svg)](https://doi.org/10.5281/zenodo.16757337)

Cross-Subject G-softmax Deep Domain Generalization Motor Imagery Classification in Brain–Computer Interfaces

## Abstract
**Background.** In cross-subject electroencephalography (EEG) motor imagery decoding tasks, significant physiological differences among individuals pose substantial challenges. Although Deep Domain Adaptation (DDA) methods have achieved considerable progress, they remain highly dependent on target domain data, which is inconsistent with real-world scenarios where target domain data may be inaccessible or extremely limited. Moreover, existing DDA methods primarily achieve feature alignment by minimizing distribution discrepancies between the source and target domains. However, given the pronounced physiological variability across individuals, simple distribution matching strategies often fail to effectively mitigate domain shift, thereby limiting generalization performance. 

**Methods.** To address these challenges, this study proposes an improved G-softmax Deep Domain Generalization (G-softmax DDG) framework, which aims to overcome the limitations of traditional DDG methods in handling inter-class differences and cross-domain distribution shifts. By introducing multi-source domain joint training and an enhanced G-softmax function, the proposed method effectively resolves the dynamic balance between intra-class distance and inter-class distance. The improved G-softmax mechanism integrates class center information, thereby enhancing model robustness and improving its ability to learn discriminative feature representations, ultimately leading to superior classification performance. 

**Results.** Experimental results demonstrate that the proposed method achieves classification performance comparable to that of DDA on two publicly available real-world EEG datasets. Moreover, it outperforms existing methods on the Lee2019_MI dataset, providing a novel solution for cross-subject motor imagery decoding. The source code is available at: https://github.com/dawin2015/G-softmax-DDG.

## G-Softmax DDG Framework
**Motivation of G-Softmax DDG.** The core idea of G-softmax is based on the assumption that the features of each class follow a Gaussian distribution in the feature space. By embedding this distributional assumption into the classification process, the model can better capture and utilize the internal structure of each class, thereby enhancing intra-class compactness.
<p align="center">
  <img width="554" height="902" alt="image" src="https://github.com/user-attachments/assets/57efe967-74df-4200-bed7-95d86cf47cd4" />
</p>

## Links to Datasets:
[BNCI2014-1/4](http://bnci-horizon-2020.eu/database/data-sets)

[Lee2019-MI](http://gigadb.org/dataset/100542)

## Citation
```bibtex
@article{liu_2025,
  author    = {Darong, Liu and Chaw Seng, Woo and Shier Nee, Saw and Yiqing He},
  title     = {Cross-Subject G-softmax Deep Domain Generalization Motor Imagery Classification in Brain–Computer Interfaces},
  journal   = {Unpublished},
  year      = {2025},
  note      = {Submitted to PeerJ CS on March,23rd,2025},
  url       = {https://doi.org/10.5281/zenodo.16757337},
  archivePrefix = {xxxx},
  eprint    = {xxxxxx},
}
```
