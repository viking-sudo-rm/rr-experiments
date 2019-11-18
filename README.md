# rr-experiments
Experiments with rational recurrences, pooling functions, and absolute values.

## Usage

```shell
pip install allennlp
CUDA_VISIBLE_DEVICES=0 RNN=lstm allennlp train configs/max_difference.jsonnet \
    -s /tmp/lstm --include-package rr_experiments
```