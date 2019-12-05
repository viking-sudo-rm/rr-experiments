# rr-experiments
Experiments with rational recurrences, pooling functions, and absolute values.

## Usage

### Installation

Built using allennlp. We use the fastai QRNN implementation.

```shell
pip install allennlp
pip install fastai
```

For some reason, jsonnet isn't supported on Windows, so we recommend running this on Mac/Linux so that the config macros work properly.

### Training

```shell
RNN=lstm LAYERS=1 allennlp train configs/max_difference.jsonnet \
    -s /tmp/lstm --include-package rr_experiments
```

The RNN architecture is specified by `RNN`. Some example options are `rnn`, `gru`, `lstm`, and `qrnn`. It's pretty easy to define and register your own `Seq2SeqEncoder` to slot in here. The variable `LAYERS` specifies how many RNN layers to use.

### Evaluation

To evaluate on 500 sentences of length 2048:

```shell
allennlp evaluate /tmp/lstm/model.tar.gz 500:2048 --include-package rr_experiments
```

To disable CUDA, set `cuda_device: -1` in configs/max_difference.jsonnet.