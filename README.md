# SSCNN

Welcome to the official repository of the SSCNN paper: [Parsimony or Capability? Decomposition Delivers Both in Long-term Time Series Forecasting](https://openreview.net/pdf?id=wiEHZSV15I).

[[Poster]](https://nips.cc/media/PosterPDFs/NeurIPS%202024/93133.png?t=1730630856.8418543)

## Model Implementation

SSCNN’s performance can be influenced by the connection layout of the four normalization blocks at each layer. We provide two connection strategies: **series** and **parallel**. Specify the desired connection by setting the `connection` parameter to `"series"` or `"parallel"` in the script.

Each normalization block or attention operation can also be enabled or disabled by setting its respective parameter to `1` (enabled) or `0` (disabled), providing flexibility for experimentation.

## Getting Started

### Environment Requirements
To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:
```
conda create -n SSCNN python=3.8
conda activate SSCNN
pip install -r requirements.txt
```

### Data Preparation
All the datasets needed for SSCNN can be obtained from the Google Drive provided in Autoformer. Create a separate folder named ./dataset and place all the CSV files in this directory. Note: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### Training Example
You can specify separate scripts to run independent tasks, such as obtaining results on electricity:
```
sh scripts/long_term_forecast/ECL_script/SSCNN.sh
```
> [!NOTE]
> We maintain a constant learning rate throughout training. Therefore, the adjust_learning_rate function from Time-Series-Library is omitted. This adjustment is critical for reproducing the reported outcomes. The default learning rate is set to 0.0005 (as per Time-Series-Library), but increasing it may help accelerate convergence.

## Citation

```
@inproceedings{deng2024parsimony,
  title={Parsimony or Capability? Decomposition Delivers Both in Long-term Time Series Forecasting},
  author={Deng, Jinliang and Ye, Feiyang and Yin, Du and Song, Xuan and Tsang, Ivor and Xiong, Hui},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/yuqinie98/patchtst

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Time-Series-Library

