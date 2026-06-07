import dataclasses

@dataclasses.dataclass
class Config:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001

    def __post_init__(self):
        self.batch_size = self.batch_size
        self.num_epochs = self.num_epochs
        self.learning_rate = self.learning_rate