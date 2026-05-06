import abc
import collections.abc
import functools

import torch

from simpoly.vivace import keys

from . import neighborhood
from .datapoint import MLFFDatapoint


def _convert_to_dtype(s: str | torch.dtype) -> torch.dtype:
    name_to_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if isinstance(s, torch.dtype):
        return s
    return name_to_dtype[s]


class Transform(abc.ABC):
    """A callable MLFFDatapoint->MLFFDatapoint transformation function
    and a string identifier bundled."""

    @abc.abstractmethod
    def __call__(self, data: MLFFDatapoint) -> MLFFDatapoint:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class ComposedTransform(Transform):
    r"""Composes multiple transforms sequentially. Supports list-style and variadic-style calling.
    The new identifier will be the concatenation of provided transforms."""

    def __init__(
        self,
        t: (
            Transform
            | collections.abc.Mapping[str, Transform]
            | collections.abc.Iterable[Transform]
        ),
        *t_rest: Transform,
    ) -> None:
        super().__init__()
        if isinstance(t, Transform):
            self.ts = [t]
        elif isinstance(t, collections.abc.Mapping):
            self.ts = [t[k] for k in t]  # pyright: ignore
        elif isinstance(t, collections.abc.Iterable):
            self.ts = [t_ for t_ in t]  # type: ignore
        else:
            raise ValueError(f"Invalid type for t: {type(t)}")

        assert all(
            isinstance(t, Transform) for t in self.ts
        ), f"All elements must be Transform, but got {self.ts}"

        self.ts += t_rest

    def __str__(self) -> str:
        transform_strs = [str(t) for t in self.ts]
        transforms_str = ", ".join(transform_strs)
        return f"{self.__class__.__name__}({transforms_str})"

    def __call__(self, data: MLFFDatapoint) -> MLFFDatapoint:
        return functools.reduce(lambda current, f: f(current), self.ts, data)


class DropAllMetadata(Transform):
    def __str__(self) -> str:
        return "drop_all_metadata"

    def __call__(self, data: MLFFDatapoint) -> MLFFDatapoint:
        valid_keys = data.recognized_keys
        for k in data.keys():
            if k not in valid_keys:
                data.pop(k)
        return data


def is_float_type(t: torch.Tensor) -> bool:
    float_types = {torch.float32, torch.float64}
    return t.dtype in float_types


class DataTypeTransform(Transform):
    def __init__(self, dtype: str | torch.dtype) -> None:
        super().__init__()
        self.dtype = _convert_to_dtype(dtype)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.dtype})"

    def __call__(self, data: MLFFDatapoint) -> MLFFDatapoint:
        for k in data.keys():
            if is_float_type(data[k]):
                data[k] = data[k].to(self.dtype)
        return data


class NeighborhoodTransform(Transform):
    """Compute neighborhood list for system considering periodic boundary conditions"""

    def __init__(
        self,
        cutoff_radius: float,  # interatomic distance in the units of the datapoint's positions
        keep_edge_length: bool = False,
        order_by_distance: bool = True,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.keep_edge_length = keep_edge_length
        self.order_by_distance = order_by_distance

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(cutoff_radius={self.cutoff_radius}, keep_edge_length={self.keep_edge_length})"

    def __call__(self, data: MLFFDatapoint) -> MLFFDatapoint:
        device = data.pos.device
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
        if num_graphs > 1:
            n_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]  # [num_graphs]
        else:
            n_nodes_per_graph = torch.tensor([data.pos.shape[0]], device=device)  # [1]

        has_pbc = hasattr(data, keys.PBC)
        has_cell = hasattr(data, keys.CELL)

        if has_pbc and has_cell:
            # Nothing to do
            pass
        elif not has_pbc and not has_cell:
            # Dummy cell will have no effect on neighborhood list since pbc is all false,
            # however, the cell's volume cannot be zero.
            data.pbc = torch.tensor(
                [[False, False, False]], dtype=torch.bool, device=device
            ).expand(
                num_graphs, 3
            )  # [num_graphs, 3]
            data.cell = torch.eye(3, dtype=data.pos.dtype, device=device).expand(
                num_graphs, 3, 3
            )  # [num_graphs, 3, 3]
        else:
            raise RuntimeError("Data point has to have PBCs and a unit cell defined, or neither.")

        # We have to add an empty batch dimension to some parameters
        radius_graph_data = neighborhood.get_radius_graph(
            pos=data.pos,  # [n_atoms, 3]
            n_nodes_per_graph=n_nodes_per_graph,  # [num_graphs]
            pbc=data.pbc,  # [num_graphs, 3]
            cell=data.cell,  # [num_graphs, 3, 3]
            cutoff_radius=self.cutoff_radius,
            return_dist=self.keep_edge_length,
            order_by_distance=self.order_by_distance,
        )

        assert radius_graph_data.n_edges_per_graph.sum() == radius_graph_data.edge_index.shape[-1]

        # Update data
        # Note: we cannot put senders and receivers into the data dict, because
        # the collate function will not adapt it. Apparently, that only works for the edge_index.
        # For this reason, we do that in the forward function of the model, if in "batch" mode.
        setattr(data, keys.EDGE_INDEX, radius_graph_data.edge_index)
        setattr(data, keys.EDGE_CELL_SHIFT, radius_graph_data.cell_offsets)
        setattr(data, keys.N_EDGES_PER_GRAPH, radius_graph_data.n_edges_per_graph)
        setattr(
            data, keys.EDGE_LENGTH, radius_graph_data.edge_length if self.keep_edge_length else None
        )

        return data
