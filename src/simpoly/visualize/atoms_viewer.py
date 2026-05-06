# SPDX-License-Identifier: MIT

import collections.abc
import copy
import io
import pathlib
import typing as ty

import ase.cell
import ase.io
import dash
import numpy as np
import numpy.typing as npt

_show_labels_value = "show_labels"
_show_forces_value = "show_forces"
_wrap_atoms_value = "wrap_atoms"
_show_cell_value = "show_cell"
_zoom_value = "zoom"

_default_style = {"height": "400px", "width": "100%"}

_forces_key = "forces"


def _vec3d_to_json(v: npt.NDArray[np.float32]) -> dict[str, float]:
    assert v.shape == (3,)
    c = v.astype(float)
    return {"x": c[0], "y": c[1], "z": c[2]}


def _cell_to_json(cell: ase.cell.Cell) -> list[list[float]]:
    m = cell.array
    assert m.shape == (3, 3)
    return m.astype(float).tolist()  # type: ignore


def _atoms_has_cell(atoms: ase.Atoms) -> bool:
    # The "empty" cell in ase.Atoms is all zeros
    return not np.allclose(atoms.cell.array, np.zeros((3, 3)))


def _atoms_has_forces(atoms: ase.Atoms) -> bool:
    return atoms.calc is not None and _forces_key in atoms.calc.results


def build_checklist_options(
    disable_show_forces: bool,
    disable_show_cell: bool,
    disable_wrap_atoms: bool,
) -> list[dash.dcc.Checklist.Options]:
    return [
        {
            "label": "Show labels",
            "value": _show_labels_value,
            "disabled": False,
        },
        {
            "label": "Show forces",
            "value": _show_forces_value,
            "disabled": disable_show_forces,
        },
        {
            "label": "Show unit cell",
            "value": _show_cell_value,
            "disabled": disable_show_cell,
        },
        {
            "label": "Wrap atoms",
            "value": _wrap_atoms_value,
            "disabled": disable_wrap_atoms,
        },
        {
            "label": "Zoom",
            "value": _zoom_value,
            "disabled": False,
        },
    ]


def view(
    atoms: ase.Atoms | collections.abc.Sequence[ase.Atoms],
    style: dict[str, ty.Any] | None = None,
    port: int | str = 8050,
) -> None:
    assets_dir = pathlib.Path(__file__).absolute().parent / "dash_assets"
    app = dash.Dash(
        name=__name__,
        title="AtomsViewer",
        assets_folder=str(assets_dir),
        external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.3/3Dmol-min.js"],
        prevent_initial_callbacks=False,
    )

    if style is None:
        style = _default_style

    atoms_list: collections.abc.Sequence[ase.Atoms]
    if isinstance(atoms, ase.Atoms):
        atoms_list = [atoms]
    else:
        atoms_list = atoms
    assert len(atoms_list) > 0, "List of atoms is empty"

    mol_div_id = "mol-div-id"

    slider_id = "idx-slider-id"
    checklist_id = "checklist-id"

    id_store_id = "id-store-id"
    config_store_id = "config-store-id"

    output_store_id = "output-store-id"

    def create_layout() -> dash.html.Div:
        slider = dash.dcc.Slider(
            id=slider_id,
            value=0,
            min=0,
            max=len(atoms_list) - 1,
            step=1,
            updatemode="drag",
            marks=None,
            tooltip={"placement": "bottom", "always_visible": len(atoms_list) > 1},
            disabled=len(atoms_list) == 1,
        )
        checklist = dash.dcc.Checklist(
            id=checklist_id,
            options=build_checklist_options(  # type: ignore[arg-type]
                disable_show_forces=False,
                disable_wrap_atoms=False,
                disable_show_cell=False,
            ),
            value=[_zoom_value],  # ticked values
        )

        return dash.html.Div(
            [
                # View
                dash.html.Div(id=mol_div_id, style=style),
                dash.html.Div(
                    [
                        dash.html.Div([slider]),
                        dash.html.Div([checklist], className="checklist-container"),
                    ]
                ),
                # Storage
                dash.dcc.Store(id=id_store_id, data=mol_div_id),
                dash.dcc.Store(id=config_store_id),
                dash.dcc.Store(id=output_store_id),
            ],
        )

    @app.callback(
        inputs=[
            dash.Input(slider_id, "value"),
            dash.Input(checklist_id, "value"),
        ],
        output=dash.Output(checklist_id, "value"),
    )
    def update_checklist_value(
        idx: int,
        checklist_value: list[str],
    ) -> list[str]:
        atoms = atoms_list[idx]

        if not _atoms_has_forces(atoms) and _show_forces_value in checklist_value:
            checklist_value.remove(_show_forces_value)  # in-place removal

        if not _atoms_has_cell(atoms) and _wrap_atoms_value in checklist_value:
            checklist_value.remove(_wrap_atoms_value)  # in-place removal

        return checklist_value

    @app.callback(
        inputs=[
            dash.Input(slider_id, "value"),
        ],
        output=dash.Output(checklist_id, "options"),
    )
    def update_checklist_options(idx: int) -> list[dash.dcc.Checklist.Options]:
        atoms = atoms_list[idx]
        has_cell = _atoms_has_cell(atoms)
        return build_checklist_options(
            disable_show_forces=not _atoms_has_forces(atoms),
            disable_wrap_atoms=not has_cell,
            disable_show_cell=not has_cell,
        )

    @app.callback(
        inputs=[
            dash.Input(slider_id, "value"),
            dash.Input(checklist_id, "value"),
        ],
        output=dash.Output(config_store_id, "data"),
    )
    def build_config(
        idx: int,
        checklist_value: list[str],
    ) -> dict[str, ty.Any]:
        atoms = atoms_list[idx]
        if _wrap_atoms_value in checklist_value:
            # Note: atoms.copy does not copy over the calculator
            assert _atoms_has_cell(atoms), "Cannot wrap atoms without a cell"
            atoms = copy.deepcopy(atoms)
            atoms.wrap()  # type: ignore

        str_io = io.StringIO()
        ase.io.write(str_io, atoms, format="xyz")
        data = str_io.getvalue()

        labels = []
        if _show_labels_value in checklist_value:
            labels += [
                {
                    "text": str(i),
                    "options": {},
                    "sel": {"serial": i},  # or None
                    "noshow": True,
                }
                for i in range(len(atoms))
            ]

        arrows = []
        if _show_forces_value in checklist_value:
            positions = atoms.positions

            assert atoms.calc is not None
            forces = atoms.calc.results.get(_forces_key, None)
            if forces is not None:
                arrows += [
                    {
                        "color": "black",
                        "alpha": 0.9,
                        "radius": 0.1,
                        "mid": 0.75,
                        "start": _vec3d_to_json(positions[i]),
                        "end": _vec3d_to_json(positions[i] + forces[i]),
                    }
                    for i in range(len(atoms))
                ]

        cell_json = _cell_to_json(atoms.get_cell()) if _atoms_has_cell(atoms) else None  # type: ignore
        config = {
            "id": "mol-div",
            "viewer": {},
            "model": {
                "data": data,
                "cell": cell_json,
                "format": "xyz",  # don't use CIF
                "show_cell": _show_cell_value in checklist_value,
            },
            "styles": [
                {"sel": {}, "style": {"stick": {}}},
                {"sel": {}, "style": {"sphere": {"scale": 0.3}}},
            ],
            "labels": labels,
            "arrows": arrows,
            "lines": [],
            "zoom": _zoom_value in checklist_value,
        }

        return config

    # JS function
    app.clientside_callback(  # type: ignore[no-untyped-call]
        dash.ClientsideFunction(
            namespace="clientside",
            function_name="update_viewer",
        ),
        inputs=[
            dash.Input(id_store_id, "data"),
            dash.Input(config_store_id, "data"),
        ],
        output=[dash.Output(output_store_id, "data")],
    )

    app.layout = create_layout

    app.run(
        host="localhost",
        port=str(port),
        debug=False,
        use_reloader=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        jupyter_mode="external",
    )
