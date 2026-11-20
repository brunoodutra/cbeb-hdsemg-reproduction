"""Window normalization used by the 2D CNN in the paper.

A window is shaped ``(channels, samples)`` = ``(128, 256)`` before this
function, or a batch ``(batch, 128, 256)``. After min-max, the network
reshapes each window to ``(16, 8, 256)``.

``bychannel`` scales each electrode on its own (axis of the samples).
``bysample`` scales each time step across electrodes.
"""
import sys

import numpy as np

# Class ids as stored in each public dataset. The training code remaps them
# with np.sort before the softmax, so label 0 in the model is the smallest id.
GESTURES_65HD = [0, 10, 8, 6, 4, 2, 13, 14, 15, 16, 61, 58]
GESTURES_HYSER = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 30, 31]

WINDOW_SAMPLES = 256
N_CHANNELS = 128
GRID_HEIGHT = 8
GRID_WIDTH = 16


def normalize_windows(batch_windows, normalization_mode):
    """Min-max scale a batch of EMG windows to [0, 1].

    Args:
        batch_windows: array ``(batch, channels, samples)``.
        normalization_mode: ``"bychannel"`` or ``"bysample"``.

    Returns:
        float32 array with the same shape.
    """
    axis_map = {"bychannel": 2, "bysample": 1}
    normalization_axis = axis_map[normalization_mode]
    samples_min = batch_windows.min(axis=normalization_axis, keepdims=True)
    normalized = batch_windows - samples_min
    samples_max = normalized.max(axis=normalization_axis, keepdims=True) + sys.float_info.epsilon
    return (normalized / samples_max).astype(np.float32, copy=False)


def as_network_input(windows):
    """Reshape ``(batch, 128, 256)`` into ``(batch, 16, 8, 256)``."""
    batch = windows.shape[0]
    return windows.reshape(batch, GRID_WIDTH, GRID_HEIGHT, WINDOW_SAMPLES)
