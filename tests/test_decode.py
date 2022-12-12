import pytest
import paddle
import paddlecrepe


###############################################################################
# Test decode.py
###############################################################################


@pytest.mark.skipif(not paddle.device.is_compiled_with_cuda(), reason="Requires CUDA device")
def test_weighted_argmax_decode():
    """Tests that weighted argmax decode works without CUDA assertion error"""
    fake_logits = paddle.rand(8, 360, 128)
    decoded = paddlecrepe.decode.weighted_argmax(fake_logits)
