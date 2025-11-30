"""
Utilities and classes for interacting with MuJoCo benchmarks, managing Apptainer instances, and handling
predefined tasks like Hopper, Swimmer, Walker, and more.
"""

import re
import socket
import subprocess
import time
import xmlrpc.client
from contextlib import closing

import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor


def find_free_port():
    """
    Finds a free port available locally for network use.

    Returns:
        int: An available port number on the machine.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_apptainer_instance(port):
    """
    Checks periodically until the Apptainer instance becomes available at the specified port.

    Parameters:
        port (int): Port of the Apptainer instance to ping.
    """
    while True:
        try:
            server = xmlrpc.client.ServerProxy(f"http://localhost:{port}")
            server.ping()
            break

        except (xmlrpc.client.Fault, xmlrpc.client.ProtocolError):
            # print(f"Error occurred: {e}. Retrying...")
            pass
        except Exception:
            # print(f"Connection failed: {e}. Retrying...")
            pass
        time.sleep(1)


def create_apptainer_instance(apptainer_sif_path: str):
    """
    Starts a new Apptainer instance or attaches to an existing one.

    Parameters:
        apptainer_sif_path (str): Path to the Apptainer SIF file.

    Returns:
        int: Port number for the instance
    """
    port = find_free_port()
    result_list = subprocess.run(["apptainer", "instance", "list"], capture_output=True, text=True, check=False)
    filtered_lines = [line for line in result_list.stdout.splitlines() if "mujoco" in line]
    count = len(filtered_lines)
    if count == 0:
        subprocess.run(f"apptainer instance start {apptainer_sif_path} mujoco_{port} --port {port}".split())

        wait_for_apptainer_instance(port)
    elif count >= 1:
        # Attach to running instance by extracting the port
        instance_name = re.findall(r"\b\w+\b", filtered_lines[0])[0]
        port = int(instance_name.split("_")[1])

    return port


class MujocoFunction(BaseTestProblem):
    """
    Represents a function interfacing with the MuJoCo benchmark environment.

    This class is primarily designed to interact with MuJoCo simulation tasks by utilizing
    a subprocess for executing benchmarks with the provided task ID. The class defines methods
    to evaluate input tensors against the specified MuJoCo task and can handle noise, negation,
    and custom bounds for simulations.
    """

    def __init__(
        self,
        benchmark: str,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.benchmark = benchmark
        if mujoco_binary_path is not None:
            self.evaluation = "binary"
            self.mujoco_binary_path = mujoco_binary_path
        elif mujoco_apptainer_path is not None:
            self.evaluation = "apptainer"
            self.apptainer_sif_path = mujoco_apptainer_path
            self.port = create_apptainer_instance(mujoco_apptainer_path)
        else:
            raise ValueError(
                "To run MuJoCo benchmarks, either mujoco_binary_path or mujoco_apptainer_path must be provided."
            )

        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if self.evaluation == "binary":
            result = subprocess.run(
                [self.mujoco_binary_path, "--benchmark", self.benchmark, "-X", f"'{X.cpu().tolist()}'"],
                capture_output=True,
                text=True,
            )
            y = torch.tensor(eval(result.stdout)).to(X)
        elif self.evaluation == "apptainer":
            server = xmlrpc.client.ServerProxy(f"http://localhost:{self.port}")
            y = server.eval(self.benchmark, X.cpu().tolist())

        return Tensor(y).to(X)


class Swimmer(MujocoFunction):
    """
    16D Swimmer task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 16
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="swimmer",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )


class Hopper(MujocoFunction):
    """
    33D Swimmer task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 33
        self._bounds = [(-1.4, 1.4) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="hopper",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )


class Walker(MujocoFunction):
    """
    102D Humanoid task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 102
        self._bounds = [(-1.8, 0.9) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="walker",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )


class Cheetah(MujocoFunction):
    """
    102D Humanoid task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 102
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="cheetah",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )


class Ant(MujocoFunction):
    """
    888D Ant task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 888
        self._bounds = [(-1, 1) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="ant",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )


class Humanoid(MujocoFunction):
    """
    888D Humanoid task
    """

    PORT = None

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        mujoco_binary_path: str = None,
        mujoco_apptainer_path: str = None,
    ) -> None:
        self.dim = 6392
        self._bounds = [(-1, 1) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))

        super().__init__(
            benchmark="humanoid",
            noise_std=noise_std,
            negate=negate,
            mujoco_binary_path=mujoco_binary_path,
            mujoco_apptainer_path=mujoco_apptainer_path,
        )
