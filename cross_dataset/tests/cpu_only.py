"""CPU test guard. Does not alter any production model or loader implementation."""

import os
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


class CPUOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError("Tests must start without CUDA initialization")
        cls.guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden in CPU tests"))
        cls.guard.start()
        cls.addClassCleanup(cls.guard.stop)

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError("Unexpected CUDA initialization")
