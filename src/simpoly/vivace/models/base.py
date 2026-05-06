import dataclasses
import typing as ty

import torch.nn


@dataclasses.dataclass
class MLFFClassMetadata:
    is_edge_centered: bool
    interface: ty.Literal["mliap", "cpp"] = "mliap"


@dataclasses.dataclass
class MLFFInstanceMetadata:
    r_max: float
    lammps_cutoff: float
    dtype: torch.dtype = torch.get_default_dtype()


class MLFFModel(torch.nn.Module):
    def get_instance_metadata(self) -> MLFFInstanceMetadata:
        raise NotImplementedError

    @classmethod
    def get_cls_metadata(cls) -> MLFFClassMetadata:
        raise NotImplementedError


class MLFFMultiHeadedModel(torch.nn.Module):
    def get_instance_metadata(self) -> MLFFInstanceMetadata:
        raise NotImplementedError

    @classmethod
    def get_cls_metadata(cls) -> MLFFClassMetadata:
        raise NotImplementedError
