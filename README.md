# rr-experiments
Experiments with rational recurrences, pooling functions, and absolute values.

## Usage

Installation:

```shell
pip install allennlp
```

To train an LSTM:

```shell
CUDA_VISIBLE_DEVICES=0 RNN=lstm H=None allennlp train configs/max_difference.jsonnet -s /tmp/lstm --include-package rr_experiments
```

To evaluate on 500 sentences of length 2048:

```shell
CUDA_VISIBLE_DEVICES=0 allennlp evaluate /tmp/lstm/model.tar.gz 500:2048 --include-package rr_experiments
```
