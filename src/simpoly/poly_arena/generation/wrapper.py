# SPDX-License-Identifier: MIT

import dataclasses
import logging
import os
import pathlib
import signal
import subprocess
import typing as ty

import pyemc

from . import templates

LOG = logging.getLogger(__name__)


def get_emc_root_dir() -> str:
    """Absolute path to the EMC root directory, located within the pyemc package."""
    return os.path.join(os.path.dirname(pyemc.__file__), "emc")


def get_emc_setup_path() -> str:
    return os.path.join(get_emc_root_dir(), "scripts", "emc_setup.pl")


def get_emc_path() -> str:
    return os.path.join(get_emc_root_dir(), "bin", "emc_linux_x86_64")


def run_command(
    args: list[str],
    working_dir: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args=args,
        cwd=working_dir,
        capture_output=True,
        shell=False,
    )


def run_emc_setup(
    esh_file: str,
    working_dir: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    LOG.debug("Running EMC setup")
    return run_command(
        args=[get_emc_setup_path(), esh_file],
        working_dir=working_dir,
    )


def run_emc(
    esh_file: str,
    working_dir: str | None = None,
    n_threads: int | None = 1,
) -> subprocess.CompletedProcess[bytes]:
    if n_threads is None:
        n_threads = os.cpu_count()

    args = [get_emc_path(), f"-nthreads={n_threads}", esh_file]

    LOG.debug("Running EMC")
    return run_command(
        args=args,
        working_dir=working_dir,
    )


def read_emc_template() -> str:
    path = pathlib.Path(__file__).parent / "resources" / "emc_template.esh"
    return open(path).read()


def format_value(value: ty.Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


@dataclasses.dataclass
class Config:
    ru_smiles: str
    first_cap: str
    second_cap: str
    n_ru_per_chain: int
    n_total: int

    density: float
    temperature: float

    seed: int = 42
    pdb: bool = False  # Write PDB file (can lead to segfaults)

    def render(self) -> str:
        template_str = read_emc_template()
        placeholder_dict = templates.config_to_placeholder_dict(dataclasses.asdict(self))
        formatted_dict = {key: format_value(value) for key, value in placeholder_dict.items()}
        return templates.render_template(template_str, formatted_dict)


class EMCError(Exception):
    def __init__(self, return_code: int, stdout: str, stderr: str) -> None:
        super().__init__()
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(return_code={self.return_code}, stdout={self.stdout}, stderr={self.stderr})"


class SegfaultError(EMCError):
    pass


class MissingForceFieldParametersError(EMCError):
    pass


class AmbiguousChargeAssignmentError(EMCError):
    pass


class TotalChargeError(EMCError):
    pass


class SimulationError(EMCError):
    """Error during (MC) simulation in EMC"""


class SetupError(EMCError):
    pass


def identify_error(return_code: int, stdout: str, stderr: str) -> EMCError:
    ErrorClass: type[EMCError]
    if "Missing force field parameters." in stdout:
        ErrorClass = MissingForceFieldParametersError
    elif "Ambiguous charge assignments for group" in stdout:
        ErrorClass = AmbiguousChargeAssignmentError
    elif "Total charge of system 'main' does not equal zero" in stdout:
        ErrorClass = TotalChargeError
    elif "Error: core/types/inverse/angle.c:377 InverseAngleInit:" in stdout:
        ErrorClass = SimulationError  # triggered by high temperatures
    elif return_code == -signal.SIGSEGV:
        ErrorClass = SegfaultError
    else:
        ErrorClass = EMCError

    return ErrorClass(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def prepare(
    config: Config,
    directory: str,
    clean_up: bool = True,
    n_threads: int | None = 1,
) -> None:
    esh_str = config.render()
    filename_root = "build"

    # EMC setup run
    esh_path = os.path.join(directory, f"{filename_root}.esh")
    with open(esh_path, "w") as f:
        f.write(esh_str)

    setup_output = run_emc_setup(esh_path, working_dir=directory)

    if setup_output.returncode != 0:
        raise SetupError(
            return_code=setup_output.returncode,
            stdout=setup_output.stdout.decode(),
            stderr=setup_output.stderr.decode(),
        )

    # Actual EMC run
    emc_config_path = os.path.join(directory, f"{filename_root}.emc")
    emc_output = run_emc(esh_file=emc_config_path, working_dir=directory, n_threads=n_threads)

    if emc_output.returncode != 0:
        raise identify_error(
            return_code=emc_output.returncode,
            stdout=emc_output.stdout.decode(),
            stderr=emc_output.stderr.decode(),
        )

    # Cleanup
    project_name = "system"
    if clean_up:
        for suffix in [".emc.gz", ".in"]:
            file_path = os.path.join(directory, f"{project_name}{suffix}")
            if os.path.exists(file_path):
                LOG.debug(f"Removing {file_path}")
                os.remove(file_path)
            else:
                LOG.debug(f"File {file_path} does not exist, skipping removal.")
