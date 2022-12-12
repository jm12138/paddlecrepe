import os

import numpy as np
import paddle
import paddlecrepe
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # Pypaddle is not compatible with non-writeable arrays, so we make a copy
    return paddle.to_tensor(np.copy(audio))[None], sample_rate


def model(capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    paddlecrepe.infer.capacity = capacity
    paddlecrepe.infer.model = paddlecrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pdparams')
    paddlecrepe.infer.model.set_state_dict(
        paddle.load(file))

    paddlecrepe.infer.model = paddlecrepe.infer.model

    # Eval mode
    paddlecrepe.infer.model.eval()
