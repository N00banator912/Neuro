# Network Logger
# K. E. Brown, Chad GPT.
# First:    2025-10-10
# Updated:  2025-10-10

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, log_dir="logs/run"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    # --- Scalar metrics (loss, reward, etc.)
    def log_scalar(self, name, value, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)
        self.writer.flush()

    # --- Histogram logging (weights, actions, etc.)
    def log_histogram(self, name, values, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
        self.writer.flush()

    # --- Matplotlib Figure logging
    def log_figure(self, name, fig, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            tf.summary.image(name, self._figure_to_image(fig), step=step)
        self.writer.flush()

    # --- Convert matplotlib figure to TF image tensor
    def _figure_to_image(self, fig):
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        return tf.expand_dims(image, 0)

    def next_step(self):
        self.step += 1
