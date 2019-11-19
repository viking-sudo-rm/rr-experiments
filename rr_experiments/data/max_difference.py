from typing import List
from overrides import overrides
import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.fields.sequence_label_field import SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("max_difference")
class MaxDifferenceReader(DatasetReader):

    def __init__(self,
                 max_difference: int = 5,
                 seed: int = 2):
        super().__init__(lazy=False)
        self.max_difference = max_difference
        self.seed = 2
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        """In order to fit the standard API for reading data, we use a "path" to specify properties
        of the data to generate."""
        num_samples, length = [int(x) for x in path.split(":")]
        random.seed(self.seed)
        for _ in range(num_samples):
            tokens, tags = self._sample(length)
            yield self.text_to_instance(tokens, tags)

    @overrides
    def text_to_instance(self, tokens: List[str], tags: List[str]) -> Instance:
        tokens_field = TextField([Token(token) for token in tokens], self.token_indexers)
        tags_field = SequenceLabelField(tags, sequence_field=tokens_field)
        return Instance({
            "tokens": tokens_field,
            "tags": tags_field,
        })

    def _sample(self, length: int):
        tokens = []
        tags = []
        count = 0
        for _ in range(length):
            token = random.choice("ab")
            count += 1 if token == "a" else -1
            tag = abs(count) < self.max_difference
            tag = str(int(tag))
            tokens.append(token)
            tags.append(tag)
        return tokens, tags
