import hashlib
import json
import os
# from logging import getLogger
import keras.backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2

from config import Config

logger = getLogger(__name__)

class SequenceModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None

    def build(self):
        # Input for the board, hand, and opponent’s belief
        in_x = x = Input(
            (10, 10, 4)
        )  # Updated for Sequence (10x10 board + hand, discard, belief)

        # Convolutional layers for board processing
        x = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        # Dense layers for hand and discard pile processing
        y = Dense(256, activation="relu")(x)

        # Policy head (outputs probabilities for each action)
        policy_out = Dense(104, activation="softmax", name="policy_out")(y)

        # Value head (outputs game outcome prediction)
        value_out = Dense(1, activation="tanh", name="value_out")(y)

        self.model = Model(in_x, [policy_out, value_out], name="sequence_model")

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding="same",
            data_format="channels_last",
            kernel_regularizer=l2(mc.l2_reg),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding="same",
            data_format="channels_last",
            kernel_regularizer=l2(mc.l2_reg),
        )(x)
        x = BatchNormalization()(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            return True
        else:
            return False

    def save(self, config_path, weight_path):
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)

    def objective_function_for_policy(y_true, y_pred):
        return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)

    def objective_function_for_value(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)