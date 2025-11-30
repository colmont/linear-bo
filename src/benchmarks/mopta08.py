"""
Mopta08 benchmark.

Adapted from https://github.com/LeoIV/BenchSuite/blob/master/benchsuite/mopta08.py
"""

import logging
import os
import platform
import stat
import subprocess
import sys
import tempfile
import urllib
from functools import cached_property
from pathlib import Path

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from jaxtyping import Float
from torch import Tensor


class Mopta08(SyntheticTestFunction):
    """
    Mopta08 benchmark task.

    :param noise_std: Standard deviation of the Gaussian noise to be added to the function values.
    """

    def __init__(self, noise_std: float = 0.0):
        # BoTorch initialization
        self.dim = 124
        bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        super().__init__(noise_std=noise_std, negate=True, bounds=bounds)

    @cached_property
    def _mopta_exectutable(self):
        sysarch = 64 if sys.maxsize > 2**32 else 32
        machine = platform.machine().lower()

        match machine:
            case "armv7l":
                assert sysarch == 32, "Not supported"
                _mopta_exectutable_relpath = "mopta08_armhf.bin"
            case "x86_64":
                assert sysarch == 64, "Not supported"
                _mopta_exectutable_relpath = "mopta08_elf64.bin"
            case "i386":
                assert sysarch == 32, "Not supported"
                _mopta_exectutable_relpath = "mopta08_elf32.bin"
            case _:
                raise RuntimeError(f"Machine with architecture {machine} ({sysarch}) is not supported")

        _mopta_executable = os.path.join(
            Path(__file__).parent.parent.parent, "data", "mopta08", _mopta_exectutable_relpath
        )
        if not os.path.exists(_mopta_executable):
            url = f"https://github.com/LeoIV/BenchSuite/raw/6bb3a7514eeb0403b10159d9196ba42d363f4817/data/mopta08/{_mopta_exectutable_relpath}"
            logging.info(f"Downloading {url} to {_mopta_executable}")
            urllib.request.urlretrieve(url, _mopta_executable)
            permissions = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
            os.chmod(self._mopta_exectutable, permissions)
        return _mopta_executable

    def _evaluate_true(self, X: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
        """
        Evaluate Mopta08 benchmark.

        :param x: input configuration
        :return: function value
        """
        *shape, d = X.shape
        X = X.view(-1, d)
        res = torch.empty(*shape, dtype=X.dtype, device=X.device)

        for i, x in enumerate(X):
            with tempfile.TemporaryDirectory() as tmpdirname:
                # write input to file in dir
                with open(os.path.join(tmpdirname, "input.txt"), "w+") as tmp_file:
                    for _x in x:
                        tmp_file.write(f"{_x.detach().cpu().numpy()}\n")

                # pass directory as working directory to process
                popen = subprocess.Popen(self._mopta_exectutable, stdout=subprocess.PIPE, cwd=tmpdirname)
                popen.wait()
                # read and parse output file
                output = open(os.path.join(tmpdirname, "output.txt"), "r").read().split("\n")
                output = [x.strip() for x in output]
                output = torch.tensor([float(x) for x in output if len(x) > 0], dtype=X.dtype, device=X.device)
                value = output[0]
                constraints = output[1:]

                # see https://arxiv.org/pdf/2103.00349.pdf E.7
                res[i] = value + 10 * torch.sum(torch.clip(constraints, min=0, max=None))

        # Done!
        return res.view(*shape)
