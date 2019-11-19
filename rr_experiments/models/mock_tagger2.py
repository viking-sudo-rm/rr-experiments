from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from torch.nn import Parameter

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("mock_false_tagger")
class MockFalseTagger(Model):

    """Always return true or false."""

    def __init__(
        self,
        vocab: Vocabulary,
        value: bool = False,
    ) -> None:
        super().__init__(vocab)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }

        self.value = int(value)

        # Need some parameters so allennlp doesn't complain.
        self.param = Parameter(torch.tensor(5.))

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, seq_len = tags.size()
        mask = get_text_field_mask(tokens)
        logits = torch.zeros(batch_size, seq_len, 2, device=tags.device)
        logits[:, :, self.value] = 1.

        output_dict = {"logits": logits}

        if tags is not None:
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            # Can't use a leaf.
            output_dict["loss"] = 2 * self.param

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }
