# -*- coding: utf-8 -*-
import unittest

from liquiforge.tokenizers.bpe import BPETokenizer
from liquiforge.tokenizers.vision_patch import VisionPatchTokenizer, synthetic_image
from liquiforge.tokenizers.audio_spec import AudioSpecTokenizer, synthetic_wave
from liquiforge.tokenizers.unified import UnifiedTokenizer, Modality


class TestBPE(unittest.TestCase):
    def test_roundtrip(self):
        t = BPETokenizer(512)
        t.train(["hello hello world", "world hello"])
        text = "hello world"
        self.assertEqual(t.decode(t.encode(text)), text)

    def test_vocab_size(self):
        t = BPETokenizer(300)
        t.train(["abc abc abc def"])
        # merges learned = vocab_size - 256 (capped by available pairs)
        self.assertTrue(len(t.merges) <= 44)


class TestVision(unittest.TestCase):
    def test_encode_shape(self):
        tok = VisionPatchTokenizer(28, 4, 128)
        img = synthetic_image(28)
        tokens = tok.encode(img)
        self.assertEqual(len(tokens), 49)
        self.assertEqual(len(tokens[0]), 128)


class TestAudio(unittest.TestCase):
    def test_encode(self):
        tok = AudioSpecTokenizer(256, 64, 80, 256)
        ids = tok.encode(synthetic_wave(1600))
        self.assertTrue(len(ids) > 0)
        self.assertTrue(all(0 <= i < 256 for i in ids))


class TestUnified(unittest.TestCase):
    def test_global_id(self):
        u = UnifiedTokenizer(128, 256, 49, 256)
        self.assertEqual(u.global_id(Modality.TEXT, 0), 0)
        self.assertEqual(u.global_id(Modality.VISION, 0), 256)

    def test_embed_dim(self):
        u = UnifiedTokenizer(128, 256, 49, 256)
        self.assertEqual(len(u.embed(Modality.TEXT, 3)), 128)


if __name__ == "__main__":
    unittest.main()
