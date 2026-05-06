import torch_geometric.data

from .datapoint import MLFFDatapoint, get_dummy_datapoint
from .transform import NeighborhoodTransform


def datapoints_to_batch(
    datapoints: list[MLFFDatapoint],
    cutoff_radius: float,
    batch_size: int = 1,
) -> MLFFDatapoint:
    """Create a batched datapoint from a list of datapoints with neighborhood computation.

    This is a lightweight version that avoids pulling in the full dataset/dataloader machinery.
    """
    transform = NeighborhoodTransform(cutoff_radius=cutoff_radius)
    transformed = [transform(dp) for dp in datapoints]
    batch: MLFFDatapoint = torch_geometric.data.Batch.from_data_list(transformed)  # type: ignore
    return batch


def build_tracer_batch(
    cutoff_radius: float = 1.2,
    n_graphs: int = 2,
) -> MLFFDatapoint:  # actually, MLFFDatapointBatch
    datapoints = [get_dummy_datapoint() for _ in range(n_graphs)]
    return datapoints_to_batch(datapoints, cutoff_radius=cutoff_radius, batch_size=n_graphs)
