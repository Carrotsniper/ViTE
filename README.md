[![GitHub license](https://img.shields.io/github/license/Carrotsniper/ViTE)](https://github.com/Carrotsniper/ViTE/blob/main/LICENSE)
## Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction

This is the official implementation of our paper **UniEdge** [https://arxiv.org/abs/2502.02504](https://arxiv.org/abs/2502.02504).

This work has been accepted at the IEEE Transactions on Circuits and Systems for Video Technology (TCSVT 2025).

![](struct.PNG)

## Abstract
Pedestrian trajectory prediction aims to forecast future movements based on historical paths. Spatial-temporal (ST) methods often separately model spatial interactions among pedestrians and temporal dependencies of individuals. They overlook the direct impacts of interactions among different pedestrians across various time steps (i.e., high-order cross-time interactions). This limits their ability to capture ST inter-dependencies and hinders prediction performance. To address these limitations, we propose UniEdge with three major designs. Firstly, we introduce a unified ST graph data structure that simplifies high-order cross-time interactions into first-order relationships, enabling the learning of ST inter-dependencies in a single step. This avoids the information loss caused by multi-step aggregation. Secondly, traditional GNNs focus on aggregating pedestrian node features, neglecting the propagation of implicit interaction patterns encoded in edge features. We propose the Edge-to-Edge-Node-to-Node Graph Convolution (E2E-N2N-GCN), a novel dual-graph network that jointly models explicit N2N social interactions among pedestrians and implicit E2E influence propagation across these interaction patterns. Finally, to overcome the limited receptive fields and challenges in capturing long-range dependencies of auto-regressive architectures, we introduce a transformer encoder-based predictor that enables global modeling of temporal correlation. UniEdge outperforms state-of-the-arts on multiple datasets, including ETH, UCY, and SDD.

#### Steps
1. Clone the project into a local machine：
   ```bash
   git clone https://github.com/Carrotsniper/UniEdge.git
   ```
2. Enter the project and unzip dataset files：
   ```bash
   cd UniEdge
   unzip dataset.zip
   ```
3. Install environments：
   ```bash
   conda env create -f environment.yaml
   ```
4. Run testing：
   ```bash
   python test.py
   ```

### Acknowledgement
Part of our code is borrowed from [DDL](https://github.com/sydney-machine-learning/pedestrianpathprediction), [GP-GRAPH](https://github.com/InhwanBae/GPGraph), [iTransformer](https://github.com/thuml/iTransformer), 
[CaST](https://github.com/yutong-xia/CaST), [GATv2](https://github.com/tech-srl/how_attentive_are_gats). We thank the authors for releasing their code and models.

## Cite this Work

If this work is useful, please consider citing the paper, and/or mentioning this repository:
```bibtex
@article{li2025uniedge,
  title={Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction},
  author={Li, Ruochen and Qiao, Tanqiu and Katsigiannis, Stamos and Zhu, Zhanxing and Shum, Hubert PH},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```
