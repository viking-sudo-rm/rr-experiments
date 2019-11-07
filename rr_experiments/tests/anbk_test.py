from unittest import TestCase

from rr_experiments.data.anbk import AnbkReader


class TestAnbkReader(TestCase):

    def setUp(self):
        super().setUp()
        self.reader = AnbkReader()

    def test_read(self):
        sents = self.reader.read(5)
        import pdb; pdb.set_trace()