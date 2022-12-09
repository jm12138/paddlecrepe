import pytest
import paddle
import paddlecrepe


###############################################################################
# Test decode.py
###############################################################################


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="Requires CUDA device")
def test_weighted_argmax_decode():
    """Tests that weighted argmax decode works without CUDA assertion error"""
    fake_logits = paddle.rand(8, 360, 128, device="cuda")
    decoded = paddlecrepe.decode.weighted_argmax(fake_logits)
