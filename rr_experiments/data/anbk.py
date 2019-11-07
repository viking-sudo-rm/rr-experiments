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

    def __init__(self,
                 prefix: bool = True,
                 abs_value: bool = False):
        super().__init__(lazy=False)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.prefix = prefix
        self.abs_value = abs_value

    def _read(self, path: str):
        """Build string/label pairs of the form:
            aa-bbbb / 00-0110
            aaaa-bbbbbbbb / 0000-00111100

        num_sents is read as a string because of AllenNLP API.
        """
        start_n, end_n = [int(n) for n in path.split(":")]
        assert start_n % 2 == 0
        assert end_n % 2 == 0

        for n in range(start_n, end_n, 2):
            tokens, tags = [], []

            if self.prefix:
                # Add the prefix of a's.
                tokens.extend("a" for _ in range(n))
                tags.extend("0" for _ in range(n))

            tokens.extend("b" for _ in range(2 * n))
            tags.extend("0" for _ in range(n // 2))
            tags.extend("1" for _ in range(n))

            if self.abs_value:
                # Worry about switching back behavior.
                tags.extend("0" for _ in range(n // 2))
            else:
                # Don't worry about switch back behavior.
                tags.extend("1" for _ in range(n // 2))

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
