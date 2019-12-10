from typing import List
from overrides import overrides
import random

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.fields.sequence_label_field import SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("prefix")
class PrefixReader(DatasetReader):

    LEXICON = "ab"

    def __init__(self,
                 seed: int = 2):
        super().__init__(lazy=False)
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
        # Maintain a balanced distribution of positive and negative examples.
        n = random.randint(0, length // 2) if random.random() < .5 else 0

        tokens = []
        tokens.extend("a" for _ in range(n))
        tokens.extend("b" for _ in range(n))
        tokens.extend(random.choice(self.LEXICON) for _ in range(length - 2 * n))

        tags = self._get_tags(tokens)
        return tokens, tags

    @staticmethod
    def _get_tags(string):
        # Check if there is a prefix of anbn with n>0.
        tags = []
        in_prefix = True
        has_prefix = False
        counter = 0
        a_mode = True
        for char in string:
            if in_prefix:
                if a_mode:
                    if char == "a":
                        counter += 1
                    else:
                        a_mode = False
                        counter -= 1
                else:
                    if char == "b":
                        counter -= 1
                    else:
                        in_prefix = False
                        has_prefix = (counter == 0)

            tags.append(str(has_prefix))

        return tags
