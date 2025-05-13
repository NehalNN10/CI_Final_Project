from config import Config


class SequenceModelAPI:
    def __init__(self, config: Config, agent_model):
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        # Ensure the input has 4 channels (one for each feature: board, hand, discard, opponent belief)
        assert x.ndim == 4  # x should have the shape (batch_size, 10, 10, 4)
        assert x.shape[1:] == (10, 10, 4)  # 10x10 board and 4 channels for other features

        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 10, 10, 4)  # Adjust shape for Sequence input

        # Run the prediction
        policy, value = self.agent_model.model.predict_on_batch(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]  # Single sample output
        else:
            return policy, value  # Batch output