from overrides import overrides
# Need to do pip install pynvrtc==8.0 for CUDA 9.0.
# from torchqrnn import QRNN
from fastai.text.models.qrnn import QRNN

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("qrnn")
class QRNNEncoder(Seq2SeqEncoder):

    """Wraps a QRNN model into an AllenNLP encoder.
    TODO: Just use a PytorchSeq2SeqWrapper?"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 num_layers: int = 1,
                 window: int = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self._qrnn = QRNN(input_size, hidden_size,
                          batch_first=True,
                          bidirectional=bidirectional,
                          n_layers=num_layers,
                          window=window)

    @overrides
    def forward(self, inputs, mask):
        states, _ = self._qrnn(inputs, None)
        return states

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional
