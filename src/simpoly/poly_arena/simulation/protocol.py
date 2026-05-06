# SPDX-License-Identifier: MIT
import dataclasses
import os
import typing as ty
from collections.abc import Sequence

from simpoly.poly_arena import generation

from . import stages as lammps_stages
from . import tools


def get_pcff_pair_style() -> str:
    return """\
kspace_style    pppm 1e-4
include         system.params"""


def get_pcff_config() -> dict[str, ty.Any]:
    return {
        "atom_style": "full",
        "units": "real",
        "potential": get_pcff_pair_style(),
        "masses": "",  # leave this blank as for PCFF this is already defined in the data file
        "time_step": 0.5,  # fs
    }


def get_thermo_section(config: dict[str, ty.Any]) -> str:
    return """\
thermo_style    custom step temp density vol press ke pe ebond evdwl ecoul elong etotal spcpu
thermo_modify   flush yes
thermo          {n_thermo_freq}
timestep        {time_step}

fix fMOM all momentum 100 linear 1 1 1 angular
print "# poly_id: {poly_id}" file thermo_data.txt
print "# temp_target: {temp_target}" append thermo_data.txt
print "# n_atoms: {n_atoms}" append thermo_data.txt
print "# seed: {seed}" append thermo_data.txt
fix thermo_log all print {n_thermo_freq} "$(step) $(temp) $(density) $(vol) $(press) $(ke) $(pe) $(etotal)" append thermo_data.txt screen no title "step temp density vol press ke pe etotal"
""".format(**config)


def pcff_header_fn(config: dict[str, ty.Any]) -> str:
    return """
atom_style      {atom_style}
units           {units}
{start_command}       {data_file}

{potential}

neighbor        1.5 bin  # or 0.5
neigh_modify    delay 50 every 2 check yes

""".format(**config) + get_thermo_section(config)


def get_mlff_pair_style(path: str, atom_types: list[str]) -> str:
    atom_types_str = " ".join(atom_types)
    return f"pair_style mliap unified {path} 0\npair_coeff * * {atom_types_str}\n"


def get_mlff_config(
    mlff_path: str,
    data_path: str,  #  atomic, metal
    atom_types: list[str],
) -> dict[str, ty.Any]:
    potential_str = get_mlff_pair_style(
        path=mlff_path,
        atom_types=atom_types,
    )

    units = "metal"
    return {
        "atom_style": "atomic",
        "units": units,
        "data_file": data_path,
        "potential": potential_str,
        "masses": tools.get_masses_section(atom_types, units=units),
        "time_step": 0.0005,  # ps
    }


def mlff_header_fn(config: dict[str, ty.Any]) -> str:
    return """
atom_style      {atom_style}
atom_modify map array sort 0 2.0
units           {units}
newton          on
{start_command}       {data_file}
{potential}
{masses}
neighbor        0.5 bin
neigh_modify    delay 50 every 2 check yes

""".format(**config) + get_thermo_section(config)


class LAMMPSProtocol:
    def __init__(
        self,
        stages: Sequence[lammps_stages.LAMMPSStage],
        units: str,
        time_step: float,
        steps_per_dump: int = 5000,
    ):
        # Check that names are unique
        assert len(stages) == len(set(s.name for s in stages)), "Stage names must be unique"
        self.stages = stages
        self.units = units
        self.time_step = time_step
        self.steps_per_dump = steps_per_dump

    def render(self) -> str:
        blocks = [
            stage.render(
                units=self.units,
                time_step=self.time_step,
                steps_per_dump=self.steps_per_dump,
            )
            for stage in self.stages
        ]
        return "\n\n".join(blocks)


@dataclasses.dataclass(frozen=True, slots=True)
class PolymerBuildResult:
    n_tot: int
    n_ru_per_chain: int
    n_chains: int
    smiles: str
    density: float


def create_lammps_input(
    directory: str,
    smiles: str,
    end_groups: tuple[str, str],
    density: float,
    temperature: float,
    n_tot: int,
    n_ru_per_chain: int,
    seed: int,
) -> PolymerBuildResult:
    os.makedirs(directory, exist_ok=True)

    stats = tools.compute_polymer_stats(
        n_tot=n_tot,
        n_chains=None,
        ru_smiles=smiles,
        end_group_smiles=end_groups,
        n_ru_per_chain=n_ru_per_chain,
    )

    config = generation.Config(
        ru_smiles=smiles,
        first_cap=end_groups[0],
        second_cap=end_groups[1],
        n_ru_per_chain=n_ru_per_chain,
        density=density,
        temperature=temperature,
        n_total=n_tot,
        seed=seed,
    )
    generation.prepare(config, directory)

    return PolymerBuildResult(
        n_tot=n_tot,
        n_ru_per_chain=n_ru_per_chain,
        n_chains=stats.n_chains,
        smiles=smiles,
        density=density,
    )


def build_21steps_protocol(
    temp_final_k: float = 300.0,  # K
    p_final_atm: float = 1.0,  # bar - metal
    time_prefactor: float = 1.0,
    seed: int = 42,
    pressure_couple: str = "aniso",
    **kwargs: dict[str, ty.Any],
) -> LAMMPSProtocol:
    """
    Reference: Abbott, L. J.; Hart, K. E.; Colina, C. M.
    Polymatic: A Generalized Simulated Polymerization Algorithm for Amorphous Polymers.
    Theor Chem Acc 2013, 132 (3), 1334.
    https://doi.org/10.1007/s00214-013-1334-z.

    See details in Table 1
    """
    temp_max_k = min(max(temp_final_k + 100, 700), 1_000)
    p_max_atm = 49346.2  # 50_000 bar
    stages = [
        lammps_stages.Minimize("minimization"),
        lammps_stages.SetVelocity("set_velocity", temp_k=temp_max_k, seed=seed),
        lammps_stages.NVE(
            name="nve_preheat",
            time_ps=time_prefactor * 10,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step1_highT_preheat",
            time_ps=time_prefactor * 50,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step2_lowT_preheat",
            time_ps=time_prefactor * 50,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step3_upward_shaking_highP",
            time_ps=time_prefactor * 50,
            temp_k=temp_final_k,
            pressure_atm=0.02 * p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step4_upward_shaking_highT",
            time_ps=time_prefactor * 50,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step5_upward_shaking_lowT",
            time_ps=time_prefactor * 100,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step6_upward_shaking_highP",
            time_ps=time_prefactor * 50,
            temp_k=temp_final_k,
            pressure_atm=0.6 * p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step7_upward_shaking_highT",
            time_ps=time_prefactor * 50,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step8_upward_shaking_lowT",
            time_ps=time_prefactor * 100,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step9_upward_shaking_highP",
            time_ps=time_prefactor * 50,
            temp_k=temp_final_k,
            pressure_atm=p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step10_upward_shaking_highT",
            time_ps=time_prefactor * 50,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step11_upward_shaking_lowT",
            time_ps=time_prefactor * 100,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step12_downward_shaking_highP",
            time_ps=time_prefactor * 5,
            temp_k=temp_final_k,
            pressure_atm=0.5 * p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step13_downward_shaking_highT",
            time_ps=time_prefactor * 5,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step14_downward_shaking_lowT",
            time_ps=time_prefactor * 10,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step15_downward_shaking_highP",
            time_ps=time_prefactor * 5,
            temp_k=temp_final_k,
            pressure_atm=0.1 * p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step16_downward_shaking_highT",
            time_ps=time_prefactor * 5,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step17_downward_shaking_lowT",
            time_ps=time_prefactor * 10,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="step18_downward_shaking_highP",
            time_ps=time_prefactor * 5,
            temp_k=temp_final_k,
            pressure_atm=0.01 * p_max_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NVT(
            name="step19_downward_shaking_highT",
            time_ps=time_prefactor * 5,
            temp_k=temp_max_k,
        ),
        lammps_stages.NVT(
            name="step20_downward_shaking_lowT",
            time_ps=time_prefactor * 10,
            temp_k=temp_final_k,
        ),
        lammps_stages.NPT(
            name="npt_equilibration",
            time_ps=time_prefactor * 800,
            temp_k=temp_final_k,
            pressure_atm=p_final_atm,
            pressure_couple=pressure_couple,
        ),
        lammps_stages.NPT(
            name="final_npt",
            time_ps=time_prefactor * 1000,
            temp_k=temp_final_k,
            pressure_atm=p_final_atm,
            pressure_couple=pressure_couple,
        ),
    ]

    return LAMMPSProtocol(stages=stages, **kwargs)  # type: ignore
