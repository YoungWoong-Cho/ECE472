#!/bin/env python3.8

"""
Homework 1
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


@dataclass
class Data:
    """
    Class for data generation and batch sampling
    """

    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.0, 1.0, size=(self.num_samples, self.num_features))
        clean_y = tf.math.sin(2 * np.pi * self.x)
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


class Model(tf.Module):
    """
    A linear regression model with gaussian basis functions
    """

    def __init__(self, rng, num_features, num_basis_functions):
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.mu = tf.Variable(rng.uniform(shape=[self.num_basis_functions, 1]))
        self.std = tf.Variable(rng.uniform(shape=[self.num_basis_functions, 1]))
        self.w = tf.Variable(rng.normal(shape=[self.num_basis_functions, 1]))
        self.b = tf.Variable(rng.normal(shape=[1, 1]))

    def __call__(self, x):
        x = x.astype(np.float32)
        x = (x - tf.transpose(self.mu)) ** 2
        x = x / (tf.transpose(self.std) ** 2)
        x = tf.math.exp(-x)
        return tf.squeeze(x @ self.w + self.b)

    def describe(self):
        print("".ljust(10 + 7 * self.num_basis_functions, "-"))
        print(
            "".ljust(7, " ")
            + "".join([str(i).ljust(7, " ") for i in range(self.num_basis_functions)])
        )
        print("".ljust(10 + 7 * self.num_basis_functions, "-"))
        print(
            "mu".ljust(6, " ")
            + "".join(
                [str(i)[:5].ljust(7, " ") for i in self.mu.numpy().flatten().tolist()]
            )
        )
        print(
            "std".ljust(6, " ")
            + "".join(
                [str(i)[:5].ljust(7, " ") for i in self.std.numpy().flatten().tolist()]
            )
        )
        print(
            "w".ljust(6, " ")
            + "".join(
                [str(i)[:5].ljust(7, " ") for i in self.w.numpy().flatten().tolist()]
            )
        )
        print("b".ljust(6, " ") + str(self.b.numpy().squeeze().tolist())[:5])
        print("".ljust(10 + 7 * self.num_basis_functions, "-"))

    @property
    def model(self):
        return (self.mu, self.std, self.w, self.b)


font = {
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("num_basis_functions", 10, "Number of gaussian basis functions")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logging level to debug")


def gaussian(x: np.ndarray, mu, std):
    """
    x (np.ndarray) : horizontal coordinates of the data points
    mu (scalar) : mean of the gaussian curve
    std (scalar) : standard deviation of the gaussian curve

    returns:
    y (np.ndarray) : gaussian datapoint with given mu and std
    """
    x = (x - mu) ** 2
    x = x / (std**2)
    y = np.exp(-x)
    return y


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    # Generate GT data
    data = Data(
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    # Create a model with gaussian basis functions
    model = Model(
        rng=tf_rng,
        num_features=FLAGS.num_features,
        num_basis_functions=FLAGS.num_basis_functions,
    )

    # Configure training
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    bar = trange(FLAGS.num_iters)
    writer = SummaryWriter(f"{log_dir}/basis_num_{FLAGS.num_basis_functions}")
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

        if i % 10 == 0:
            writer.add_scalar(
                "MSE Loss",
                loss.numpy(),
                i,
            )

    model.describe()

    _, ax = plt.subplots(2, 1, figsize=(5, 6), dpi=400)

    [_ax.set_xlabel("x") for _ax in ax]
    [_ax.set_ylabel("y", labelpad=10) for _ax in ax]

    # Plot data points, fitting curve, and DGP graph
    xs = np.linspace(0, 1, int(1e5))
    xs = xs[:, np.newaxis]

    ax[0].set_ylim(-np.amax(data.y) * 1.5, np.amax(data.y) * 1.5)
    ax[0].set_title("Linear fit with Gaussian basis functions")
    ax[0].plot(
        np.squeeze(data.x),
        data.y,
        "o",
        xs,
        np.sin(2 * np.pi * xs),
        "--",
        xs,
        np.squeeze(model(xs)),
        "-",
    )

    # Plot Gaussian basis functions
    xs = np.linspace(np.amin(model.mu) - 0.5, np.amax(model.mu) + 0.5, int(1e6))
    xs = xs[:, np.newaxis]

    ax[1].set_xlim(xs[0], xs[-1])
    ax[1].set_ylim(0, np.amax(data.y) * 1.5)
    ax[1].set_title("Gaussian basis functions")
    [
        ax[1].plot(xs, gaussian(xs, model.mu[i], model.std[i]))
        for i in range(FLAGS.num_basis_functions)
    ]

    plt.tight_layout()
    plt.savefig(f"{output_dir}/basis_num_{FLAGS.num_basis_functions}.pdf")


if __name__ == "__main__":
    app.run(main)
