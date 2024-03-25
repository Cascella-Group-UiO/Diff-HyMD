"""Classes used to learn model parameters through MD trajectories"""

from typing import Optional

from flax import struct
from jax import Array


@struct.dataclass
class GeneralModel:
    n_types: int = struct.field(pytree_node=False, default=1)
    type_to_chi: Array = struct.field(pytree_node=False, default=None)
    self_interaction: bool = struct.field(pytree_node=False, default=False)
    chi_constraints: dict = struct.field(pytree_node=False, default_factory=dict)
    chi: Optional[Array] = None
    bonds: Optional[dict] = None  # -> bonds_2
    angles: Optional[dict] = None  # -> bonds_3
    dihedrals: Optional[dict] = None  # -> bonds_4

    def __str__(self):
        ret_str = ""
        if self.chi is not None:
            ret_str += f"chi:\t{self.chi}\n"

        # if self.chi_constraints:
        #     ret_str += f"\t\tchi_constraints:\t{self.chi_constraints}\n"

        # NOTE: not implemented
        if self.bonds:
            pass
        if self.angles:
            pass
        if self.dihedrals:
            pass

        return ret_str


@struct.dataclass
class ChiModel:
    type_to_chi: Array = struct.field(pytree_node=False)
    self_interaction: bool = struct.field(pytree_node=False, default=False)
    chi: Optional[Array] = None
