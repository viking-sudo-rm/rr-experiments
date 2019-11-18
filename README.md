# rr-experiments
Experiments with rational recurrences, pooling functions, and absolute values.

## Usage

Installation:

```shell
pip install allennlp
pip install cupy pynvrtc git+https://github.com/salesforce/pytorch-qrnn
```

To train an LSTM:

```shell
RNN=lstm H=None allennlp train configs/max_difference.jsonnet -s /tmp/lstm --include-package rr_experiments
```

To evaluate on 500 sentences of length 2048:

```shell
allennlp evaluate /tmp/lstm/model.tar.gz 500:2048 --include-package rr_experiments
```

To disable CUDA, set `cuda_device: -1` in configs/max_difference.jsonnet.