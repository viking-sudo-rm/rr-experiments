from typing import List
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.fields.sequence_label_field import SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("anbk")
class AnbkReader(DatasetReader):

    def __init__(self):
        super().__init__(lazy=False)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, max_n: int):
        """Build string/label pairs of the form:
            aaabbbbbb
            000011100
        """
        for n in range(max_n):
            start_idx = n // 2
            end_idx = (3 * n) // 2

            tokens = []
            tokens.extend("a" for _ in range(n))
            tokens.extend("b" for _ in range(2 * n))

            tags = []
            tags.extend("0" for _ in range(n))
            tags.extend("0" for _ in range(start_idx))
            tags.extend("1" for _ in range(end_idx - start_idx))
            tags.extend("0" for _ in range(n - end_idx))

            yield self.text_to_instance(tokens, tags)

    @overrides
    def text_to_instance(self, tokens: List[str], tags: List[str]) -> Instance:
        tokens_field = TextField([Token(token) for token in tokens],
                                 self._token_indexers)
        tags_field = SequenceLabelField(tags, sequence_field=tokens_field)

        return Instance({
            "tokens": tokens_field,
            "tags": tags_field,
        })
