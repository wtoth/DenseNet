from dataclasses import dataclass

@dataclass
class HyperParameters:
    num_epochs: int = None 
    batch_size: int = None 
    learning_rate: float = None 
    momentum: float = None
    weight_decay: float = None