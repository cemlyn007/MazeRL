from flax import linen as nn

class DiscreteNetwork(nn.Module):
    input_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x

    def train(self):
        pass

    def eval(self):
        pass