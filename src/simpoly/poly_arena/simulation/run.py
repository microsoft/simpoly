# SPDX-License-Identifier: MIT

import os.path

import click

from simpoly.poly_arena import experiment
from simpoly.poly_arena.simulation import protocol
from simpoly.poly_arena.simulation import tools as sim_tools


def prepare_21_simulation(
    directory: str,
    model_type: str,
    poly_id: str,
    n_atoms: int,
    temp_k: float,
    pressure_atm: float,
    mlff_path: str | None = None,
    time_prefactor: float = 1.0,
    seed: int = 42,
) -> None:
    working_dir = os.path.abspath(directory)
    os.makedirs(working_dir, exist_ok=True)
    print(f"Preparing 21-step simulation in {working_dir} for polymer {poly_id}")

    df = experiment.load_data()
    poly_data = df.loc[poly_id]

    protocol.create_lammps_input(
        directory=working_dir,
        smiles=str(poly_data["smiles"]),
        end_groups=(str(poly_data["end_group_0"]), str(poly_data["end_group_1"])),
        density=0.5,  # initial density
        temperature=temp_k,
        n_tot=n_atoms,
        n_ru_per_chain=10,
        seed=seed,
    )

    lammps_data_path = os.path.join(working_dir, "system.data")
    if model_type == "pcff":
        config = protocol.get_pcff_config()
        header_fn = protocol.pcff_header_fn
        config["data_file"] = lammps_data_path

    elif model_type == "mlff":
        if mlff_path is None:
            raise ValueError("mlff_path is required when model_type is 'mlff'")
        metal_data_path = os.path.join(working_dir, "data.lmps")
        atom_types = sim_tools.rewrite_full_to_metal_data(lammps_data_path, metal_data_path)

        config = protocol.get_mlff_config(
            mlff_path=mlff_path,
            data_path=metal_data_path,
            atom_types=atom_types,
        )
        config["data_file"] = metal_data_path
        header_fn = protocol.mlff_header_fn
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    config["start_command"] = "read_data"
    config["n_thermo_freq"] = 250
    config["poly_id"] = poly_id
    config["temp_target"] = temp_k
    config["n_atoms"] = n_atoms
    config["seed"] = seed

    sim_protocol = protocol.build_21steps_protocol(
        temp_final_k=temp_k,
        p_final_atm=pressure_atm,
        seed=seed,
        pressure_couple="aniso",
        units=config["units"],
        time_step=config["time_step"],
        time_prefactor=time_prefactor,
    )

    blocks = sim_protocol.render()
    header = header_fn(config)
    lammps_input = "\n\n".join([header, blocks])

    # Write input file to disk
    lammps_filename = "21steps.in"
    lammps_path = os.path.join(working_dir, lammps_filename)
    with open(lammps_path, "w") as f:
        f.write(lammps_input)

    print("Done")


@click.command()
@click.option(
    "--directory",
    default=None,
    type=click.Path(),
    help="Working directory for simulation files (default: {poly_id}_n{n_atoms}_T{temp_k}_s{seed})",
)
@click.option(
    "--model-type",
    default="pcff",
    type=click.Choice(["pcff", "mlff"], case_sensitive=False),
    help="Force field model type (default: pcff)",
)
@click.option(
    "--poly-id",
    required=True,
    type=str,
    help="Polymer identifier",
)
@click.option(
    "--n-atoms",
    default=100,
    type=int,
    help="Number of atoms in the system (default: 100)",
)
@click.option(
    "--temp-k",
    default=300.0,
    type=float,
    help="Temperature in Kelvin (default: 300.0)",
)
@click.option(
    "--pressure-atm",
    default=1.0,
    type=float,
    help="Pressure in atmospheres (default: 1.0)",
)
@click.option(
    "--mlff-path",
    default=None,
    type=click.Path(exists=True),
    help="Path to MLFF model (required when model-type is mlff)",
)
@click.option(
    "--time-prefactor",
    default=1.0,
    type=float,
    help="Time prefactor for simulation speed (default: 1.0, use 0.1 for 10x faster)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility (default: 42)",
)
def main(
    directory: str | None,
    model_type: str,
    poly_id: str,
    n_atoms: int,
    temp_k: float,
    pressure_atm: float,
    mlff_path: str | None,
    time_prefactor: float,
    seed: int,
) -> None:
    """Prepare a 21-step LAMMPS simulation for polymer systems."""
    # Generate default directory name if not provided
    if directory is None:
        directory = f"{poly_id}_n{n_atoms}_T{temp_k}_s{seed}"

    prepare_21_simulation(
        directory=directory,
        model_type=model_type,
        poly_id=poly_id,
        n_atoms=n_atoms,
        temp_k=temp_k,
        pressure_atm=pressure_atm,
        mlff_path=mlff_path,
        time_prefactor=time_prefactor,
        seed=seed,
    )


if __name__ == "__main__":
    main()
