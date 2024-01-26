import numpy as np
import tensorflow as tf

from .unet import UNet


class DiffWave(tf.keras.Model):
    """Code copied and modified from DiffWave: A Versatile Diffusion Model for Audio Synthesis.
    Zhifeng Kong et al., 2020.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(DiffWave, self).__init__()
        self.config = config
        self.net = UNet(config)

    def call(self, mixture):
        """Call model."""
        return self.net(mixture)

    @staticmethod
    def check_shape(data, dim):
        n = data.shape[dim]
        if n % 2 != 0:
            n = data.shape[dim] - 1
        if dim==0:
            return data[:n, :]
        else:
            return data[:, :n]

    def write(self, path, optim=None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to write.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path, optim=None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to restore.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)

        

