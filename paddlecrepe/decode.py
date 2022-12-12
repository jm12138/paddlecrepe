import librosa
import numpy as np
import paddle

import paddlecrepe


###############################################################################
# Probability sequence decoding methods
###############################################################################


def argmax(logits):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(dim=1)

    # Convert to frequency in Hz
    return bins, paddlecrepe.convert.bins_to_frequency(bins)


def weighted_argmax(logits):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(dim=1)

    # Find bounds of analysis window
    start = paddle.max(paddle.to_tensor(0), bins - 4)
    end = paddle.min(paddle.to_tensor(logits.shape[1]), bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.shape[0]):
        for time in range(logits.shape[2]):
            logits[batch, :start[batch, time], time] = -float('inf')
            logits[batch, end[batch, time]:, time] = -float('inf')

    # Construct weights
    if not hasattr(weighted_argmax, 'weights'):
        weights = paddlecrepe.convert.bins_to_cents(paddle.arange(360))
        weighted_argmax.weights = weights[None, :, None]

    weighted_argmax.weights = weighted_argmax.weights

    # Convert to probabilities
    with paddle.no_grad():
        probs = paddle.sigmoid(logits)

    # Apply weights
    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)

    # Convert to frequency in Hz
    return bins, paddlecrepe.convert.cents_to_frequency(cents)


def viterbi(logits):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition

    # Normalize logits
    with paddle.no_grad():
        probs = paddle.nn.functional.softmax(logits, dim=1)

    # Convert to numpy
    sequences = probs.cpu().numpy()

    # Perform viterbi decoding
    bins = np.array([
        librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64)
        for sequence in sequences])

    # Convert to pypaddle
    bins = paddle.to_tensor(bins)

    # Convert to frequency in Hz
    return bins, paddlecrepe.convert.bins_to_frequency(bins)
