# SPDX-License-Identifier: MIT

import abc
import dataclasses
import math

import qcelemental as qcel

RESTART_FOLDER = "restart"
DUMP_FOLDER = "dump"


def compute_num_steps(
    time_ps: float,
    time_step: float,
    units: str,
) -> int:
    if units == "real":  # time_step is in fs
        return math.ceil(time_ps * 1000 / time_step)
    elif units == "metal":  # time_step is in ps
        return math.ceil(time_ps / time_step)
    else:
        raise ValueError(f"Unknown units: {units}")


def convert_pressure(pressure_atm: float, units: str) -> float:
    if units == "real":  # requires bar
        return pressure_atm * qcel.constants.conversion_factor("atm", "bar")  # type: ignore[no-any-return]
    elif units == "metal":  # requires atm
        return pressure_atm
    else:
        raise ValueError(f"Unknown units: {units}")


@dataclasses.dataclass
class LAMMPSStage(abc.ABC):
    name: str

    @abc.abstractmethod
    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        raise NotImplementedError()

    def render_title(self) -> str:
        return f"# Stage: {self.name}"

    def render_dump(self, steps_per_dump: int) -> str:
        return f"dump 1 all custom {steps_per_dump} {DUMP_FOLDER}/{self.name}.lammpstrj.* id type xu yu zu"

    def render_undump(self) -> str:
        return "undump 1"

    def render_mkdirs(self) -> str:
        return f"shell mkdir {RESTART_FOLDER} {DUMP_FOLDER}"

    def render_restart(self) -> str:
        return f"write_restart {RESTART_FOLDER}/{self.name}.restart"


@dataclasses.dataclass
class Minimize(LAMMPSStage):
    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        commands = [
            self.render_title(),
            self.render_mkdirs(),
            self.render_dump(steps_per_dump),
            "min_style cg",
            "minimize 1e-08 1e-10 10000000 1000000000",
            self.render_undump(),
            self.render_restart(),
        ]

        return "\n".join(commands)


@dataclasses.dataclass
class SetVelocity(LAMMPSStage):
    temp_k: float
    seed: int = 4928459

    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        commands = [
            self.render_title(),
            self.render_mkdirs(),
            self.render_dump(steps_per_dump),
            f"velocity all create {self.temp_k} {self.seed} mom yes rot yes dist gaussian",
            self.render_undump(),
            self.render_restart(),
        ]

        return "\n".join(commands)


@dataclasses.dataclass
class NVE(LAMMPSStage):
    time_ps: float
    temp_k: float
    n_damping_steps: int = 100
    seed: int = 723853

    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        n_steps = compute_num_steps(self.time_ps, time_step=time_step, units=units)

        commands = [
            self.render_title(),
            self.render_mkdirs(),
            self.render_dump(steps_per_dump),
            f"fix {self.name}_0 all langevin {self.temp_k} {self.temp_k} $({self.n_damping_steps}*dt) {self.seed}",
            f"fix {self.name}_1 all nve/limit 0.1",
            f"run {n_steps}",
            f"unfix {self.name}_0",
            f"unfix {self.name}_1",
            self.render_undump(),
            self.render_restart(),
        ]

        return "\n".join(commands)


@dataclasses.dataclass
class NVT(LAMMPSStage):
    time_ps: float
    temp_k: float
    n_damping_steps: int = 100

    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        n_steps = compute_num_steps(self.time_ps, time_step=time_step, units=units)

        commands = [
            self.render_title(),
            self.render_mkdirs(),
            self.render_dump(steps_per_dump),
            f"fix {self.name} all nvt temp {self.temp_k} {self.temp_k} $({self.n_damping_steps}*dt)",
            f"run {n_steps}",
            f"unfix {self.name}",
            self.render_undump(),
            self.render_restart(),
        ]

        return "\n".join(commands)


@dataclasses.dataclass
class NPT(LAMMPSStage):
    time_ps: float
    temp_k: float
    pressure_atm: float
    pressure_couple: str
    n_damping_steps: int = 100

    def render(
        self,
        units: str,
        time_step: float,
        steps_per_dump: int,
    ) -> str:
        n_steps = compute_num_steps(self.time_ps, time_step=time_step, units=units)
        pressure = convert_pressure(self.pressure_atm, units=units)

        commands = [
            self.render_title(),
            self.render_mkdirs(),
            self.render_dump(steps_per_dump),
            f"fix {self.name} all npt temp {self.temp_k} {self.temp_k} $({self.n_damping_steps}*dt) {self.pressure_couple} {pressure:.3f} {pressure:.3f} $({self.n_damping_steps * 10}*dt)",
            f"run {n_steps}",
            f"unfix {self.name}",
            self.render_undump(),
        ]

        return "\n".join(commands)
