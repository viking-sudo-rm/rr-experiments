from unittest import TestCase

from rr_experiments.data.anbk import AnbkReader


class TestAnbkReader(TestCase):

    def setUp(self):
        super().setUp()
        self.reader = AnbkReader()

    def test_read(self):
        instances = self.reader.read("2:6")
        tokens = ["".join(tok.text for tok in inst["tokens"]) for inst in instances]
        tags = ["".join(tag for tag in inst["tags"]) for inst in instances]

        exp_tokens = ["aabbbb", "aaaabbbbbbbb"]
        exp_tags = ["000110", "000000111100"]
        
        self.assertEqual(tokens, exp_tokens)
        self.assertEqual(tags, exp_tags)
