# Decoupling Dynamic Monocular Videos for Dynamic View Synthesis
[Paper](https://arxiv.org/abs/2304.01716)
## Getting Started

### Dependencies

* Linux
* Anaconda 3
* Python 3.8
* CUDA 11.1
* RTX 3090

### Installation


```
git clone https://github.com/mengyou2/DecoulpingNeRF.git
cd DecoulpingNeRF
conda create -n denerf python=3.8
conda activate denerf
pip install -r requirements.txt

```

### Training

To train the model by running
```
python run_nerf.py --config ./configs/config_xxxx.txt 

```

### Testing

To train the model by running
```
python run_nerf.py --config ./configs/config_xxxx.txt --render_only --ft_path ./logs/xxxx/200000.tar

```
### Acknowledgments
Our code is build upon [NeRF](https://github.com/bmild/nerf), [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields) and [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF).




