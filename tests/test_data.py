# -*- coding: utf-8 -*-
import unittest

from liquiforge.data.base import DataLoader
from liquiforge.data.text_data import TextDataset
from liquiforge.data.vision_data import VisionDataset
from liquiforge.data.audio_data import AudioDataset
from liquiforge.data.omni_data import OmniPairsDataset
from liquiforge.data.decision_data import DecisionDataset


class TestData(unittest.TestCase):
    def test_text_dataset(self):
        ds = TextDataset(["hello world hello", "abc def"])
        x, y = ds[0]
        self.assertEqual(len(x), len(y))

    def test_vision_dataset(self):
        ds = VisionDataset(8)
        img, lab = ds[0]
        self.assertEqual(len(img), 28)
        self.assertTrue(0 <= lab < 10)

    def test_audio_dataset(self):
        ds = AudioDataset(8)
        wave, ref = ds[0]
        self.assertTrue(len(wave) > 0)

    def test_omni_dataset(self):
        ds = OmniPairsDataset(8)
        cap, img = ds[0]
        self.assertTrue(isinstance(cap, str))

    def test_decision_dataset(self):
        ds = DecisionDataset(8, 64, 16)
        s, a = ds[0]
        self.assertTrue(0 <= a < 16)

    def test_dataloader_batches(self):
        ds = VisionDataset(10)
        dl = DataLoader(ds, batch_size=3, shuffle=True)
        total = sum(len(b) for b in dl)
        self.assertEqual(total, 10)


if __name__ == "__main__":
    unittest.main()
