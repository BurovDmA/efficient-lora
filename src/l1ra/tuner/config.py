from dataclasses import dataclass, field

from aenum import extend_enum
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora import LoraConfig
from peft.utils import PeftType

from l1ra.tuner.model import L1RAModel

PEFT_TYPE_TO_MODEL_MAPPING.update({"L1RA": L1RAModel})
# avoid double-registering enum entry when module is imported multiple times
if not hasattr(PeftType, "L1RA"):
    extend_enum(PeftType, "L1RA", "L1RA")


@dataclass
class L1RAConfig(LoraConfig):
    """This is the configuration class to store the configuration of a [`L1RAModel`]."""

    l1ra_lambda: float = field(
        default=1e-3, metadata={"help": "The sparse l1 regularization coefficient."}
    )
    eta_c: float = field(
        default=1e-2, metadata={"help": "The decoupled learning rate for the gate vectors."}
    )
    rank_update_ratio: int = field(
        default=0.1, metadata={"help": "Ratio of training steps between each rank update."}
    )
    prune_threshold: float = field(
        default=1e-6, metadata={"help": "Threshold under which ranks are pruned."}
    )
    reassign: bool = field(default=True, metadata={"help": "Whether to reassign pruned ranks."})
    exclude_pruned: bool = field(
        default=True,
        metadata={"help": "Whether to exclude pruned adapters from rank reassignment."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.L1RA
