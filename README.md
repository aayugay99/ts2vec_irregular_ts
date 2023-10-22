# TS2Vec for irregular time series

In today’s data-driven world, irregular time series data, often presented as event sequences, is
vital in health monitoring and finance domains. These data pose challenges due to non-uniform time intervals between events, which traditional time series methods cannot handle. This project addresses this challenge by proposing modifications to the TS2Vec model, a cutting-edge representation learning technique for regular time series data, to make it suitable for irregular time series. We propose modifying the TS2Vec model by implementing dynamic temporal pooling, loss modification, and continuous kernel convolutions to fit irregular data better.

# Reproducibility

```
conda env create -f environment.yml
conda activate ts2vec
```
# Datasets and models
Datasets we used in our experiments, as well as the pretrained models' weights and the corresponding configuration files, are available [online](https://disk.yandex.ru/d/qLjmH_LHEXgIcA).
