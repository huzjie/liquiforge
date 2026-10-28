# -*- coding: utf-8 -*-
import unittest

from liquiforge.utils.rng import stable_float, stable_int, stable_vector, stable_choice
from liquiforge.utils.text import edit_distance, word_error_rate, tokenize_words
from liquiforge.utils.yamlish import yaml_loads


class TestRng(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(stable_float("k", 0, 1), stable_float("k", 0, 1))
        self.assertEqual(stable_int("i", 0, 9), stable_int("i", 0, 9))

    def test_ranges(self):
        for i in range(20):
            self.assertTrue(0.0 <= stable_float("r%d" % i) <= 1.0)
            self.assertTrue(0 <= stable_int("q%d" % i, 3, 7) <= 7)

    def test_vector_len(self):
        v = stable_vector("v", 16)
        self.assertEqual(len(v), 16)

    def test_choice_in_items(self):
        items = ["a", "b", "c"]
        self.assertIn(stable_choice("c", items), items)


class TestText(unittest.TestCase):
    def test_edit_distance_equal(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)

    def test_edit_distance_known(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_wer_zero(self):
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)

    def test_tokenize(self):
        self.assertEqual(tokenize_words("Hello, world 2026"), ["hello", "world", "2026"])


class TestYamlish(unittest.TestCase):
    def test_scalars(self):
        d = yaml_loads("a: 1\nb: 2.5\nc: true\nd: null\ne: hello\n")
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2.5)
        self.assertIs(d["c"], True)
        self.assertIsNone(d["d"])
        self.assertEqual(d["e"], "hello")

    def test_nested(self):
        d = yaml_loads("model:\n  dim: 128\n  n_layers: 6\n")
        self.assertEqual(d["model"]["dim"], 128)
        self.assertEqual(d["model"]["n_layers"], 6)

    def test_block_list(self):
        d = yaml_loads("include:\n  - a\n  - b\n  - c\n")
        self.assertEqual(d["include"], ["a", "b", "c"])

    def test_inline_flow(self):
        d = yaml_loads("ports: [80, 443]\nmap: {x: 1, y: 2}\n")
        self.assertEqual(d["ports"], [80, 443])
        self.assertEqual(d["map"], {"x": 1, "y": 2})


if __name__ == "__main__":
    unittest.main()
