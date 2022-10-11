#!/bin/env python3.8

"""
Homework 2
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from torch.utils.tensorboard import SummaryWriter

from absl import app
from absl import flags
from tqdm import trange
from itertools import product

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
    num_samples: int
    noise: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples * 2)

        theta = (
            np.random.rand(self.num_samples).reshape(-1, 1) * 3.5 * np.pi
        )  # theta ranging from 0 to 3.5pi rad

        # Class 0
        radius_0 = theta + 1
        x_0 = np.hstack((-radius_0 * np.cos(theta), radius_0 * np.sin(theta)))
        x_0 += rng.normal(scale=self.noise, size=[self.num_samples, 2])  # add noise
        y_0 = np.zeros((self.num_samples, 1))

        # Class 1
        radius_1 = -theta - 1
        x_1 = np.hstack((-radius_1 * np.cos(theta), radius_1 * np.sin(theta)))
        x_1 += rng.normal(scale=self.noise, size=[self.num_samples, 2])  # add noise
        y_1 = np.ones((self.num_samples, 1))

        self.x = np.vstack((x_0, x_1))
        self.y = np.vstack((y_0, y_1))

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices]


class MLP(tf.Module):
    """
    MLP model for binary classification
    There were several attemps to come up with a functioning MLP network.
    The most difficult part was about the weight initialization of w and b.
    A uniform distribution between 0 and 1 resulted in predictions that are
    close to 1.0. I happened to realize that the keras Dense layer makes use
    of Gloro uniform initialization for the kernels and zero initializations
    for the biases. Also, during the training phase, the learning rate and
    the regularization coefficient tend to show dependency; in other word,
    changing learning rate required changing the value of the regularization
    coefficient as well. This induced a lot of hyperparameter tuning
    """

    def __init__(self, rng, units):
        self.rng = rng
        self.units = units

        # Weight initialization followed the default implementation of keras Dense layer
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
        self.w = [
            tf.Variable(
                self.rng.uniform(
                    shape=[self.units[i], self.units[i + 1]],
                    minval=-tf.math.sqrt(6 / (self.units[i] + self.units[i + 1])),
                    maxval=tf.math.sqrt(6 / (self.units[i] + self.units[i + 1])),
                )
            )
            for i in range(len(self.units) - 1)
        ]
        self.b = [
            tf.Variable(tf.zeros(shape=[self.w[i].shape[1]]))
            for i in range(len(self.units) - 1)
        ]

    def __call__(self, x):
        for i in range(len(self.w) - 1):
            x = tf.nn.relu(x @ self.w[i] + self.b[i])
        x = tf.nn.sigmoid(x @ self.w[-1] + self.b[-1])
        return x


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 400, "Number of samples in dataset")
flags.DEFINE_float("noise", 0.1, "Noise to be added to the spiral datapoint")
flags.DEFINE_integer("batch_size", 150, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 3000, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_float("l2_coeff", 0.001, "Coefficient for L2 regularization")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_bool("debug", False, "Set logging level to debug")

plt.rcParams["figure.figsize"] = (6, 6)


def plot(x, y=None, model=None, img_size=512, scaling_factor=14):
    """
    x: (np.ndarray) Data array of shape (N, 2) where
        first and second column is horizontal and
        vertical axis of the datapoint respectively
    y: (np.ndarray) Label array of shape (N, 2)
    model: (tf.Module) Model to predict y_hat
    img_size: (int) Image size. Defaults to 512
    scaling_factor: (int) Scaling factor from image coordinate
        to cartesian coordinate. Defaults to 14
    """
    # Produce heatmap that indicate how close
    # the predicted y_hat is to 0.5
    if model is not None:
        y = model(x)
        coords = np.array(list(product(list(range(img_size)), list(range(img_size)))))
        coords = coords * (scaling_factor * 2) / (img_size - 1) - scaling_factor
        y_hat = model(coords)
        heatmap = (
            ((-np.abs(y_hat.numpy() - 0.5) + 0.5) * 2).reshape(img_size, img_size).T
        )
        plt.imshow(
            heatmap,
            extent=(-scaling_factor, scaling_factor, -scaling_factor, scaling_factor),
            origin="lower",
        )

    # Scatter plot for the datapoints
    x0 = x[np.where(y < 0.5)[0]]
    x1 = x[np.where(y >= 0.5)[0]]
    plt.scatter(x0[:, 0], x0[:, 1], color="r", edgecolors="black")
    plt.scatter(x1[:, 0], x1[:, 1], color="b", edgecolors="black")
    plt.title("Spirals with heatmap showing p(t=1|x)=0.5")

    plt.savefig("plot.pdf")


def BCELoss(y, y_hat):
    loss = tf.reduce_mean(-y * tf.math.log(y_hat) - (1 - y) * tf.math.log(1 - y_hat))
    return loss


def L2Regularization(coeff, model):
    loss_l2 = coeff * tf.math.reduce_sum(
        [
            # Add small number in order to avoid division by zero
            # https://datascience.stackexchange.com/questions/80898/tensorflow-gradient-returns-nan-or-inf
            tf.sqrt(tf.reduce_sum(tf.square(var)) + 1.0e-12)
            for var in model.trainable_variables
        ]
    )
    return loss_l2


def main(a):
    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    # Generate GT data
    data = Data(
        rng=np_rng,
        num_samples=FLAGS.num_samples,
        noise=FLAGS.noise,
    )

    # Create a MLP model
    model = MLP(rng=tf_rng, units=[2, 16, 16, 16, 1])

    # Configure training
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)
    writer = SummaryWriter(f"{log_dir}")
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = BCELoss(y, y_hat) + L2Regularization(FLAGS.l2_coeff, model)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

        if i % 10 == 0:
            writer.add_scalar(
                "BCE Loss",
                loss.numpy(),
                i,
            )

    plot(data.x, model=model)


if __name__ == "__main__":
    app.run(main)
