from overrides import overrides
from torchqrnn import QRNN
import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("qrnn")
class QRNNEncoder(Seq2SeqEncoder):

    """Wraps a QRNN model into an AllenNLP encoder with the same API as LSTM."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = False
        self._qrnn = QRNN(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)

    @overrides
    def forward(self, inputs, mask):
        # The Salesforce implementation doesn't support batch_first.
        inputs = torch.transpose(inputs, 0, 1)
        states, _ = self._qrnn(inputs)
        states = torch.transpose(states, 0, 1)
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
