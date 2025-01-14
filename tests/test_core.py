import numpy as np
import paddlecrepe


###############################################################################
# Test core.py
###############################################################################


def test_embed_tiny(audio):
    """Tests that the embedding is the expected size"""
    embedding = paddlecrepe.embed(audio, paddlecrepe.SAMPLE_RATE, 160, 'tiny')
    assert embedding.shape == [1, 1001, 32, 8]


def test_embed_full(audio):
    """Tests that the embedding is the expected size"""
    embedding = paddlecrepe.embed(audio, paddlecrepe.SAMPLE_RATE, 160, 'full')
    assert embedding.shape == [1, 1001, 32, 64]


def test_infer_tiny(frames, activation_tiny):
    """Test that inference is the same as the original crepe"""
    activation = paddlecrepe.infer(frames, 'tiny').detach().numpy()
    diff = np.abs(activation - activation_tiny)
    assert diff.max() < 1e-5 and diff.mean() < 1e-7


def test_infer_full(frames, activation_full):
    """Test that inference is the same as the original crepe"""
    activation = paddlecrepe.infer(frames, 'full').detach().numpy()
    diff = np.abs(activation - activation_full)
    assert diff.max() < 1e-5 and diff.mean() < 1e-7
