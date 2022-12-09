import paddle

import paddlecrepe


###############################################################################
# Test threshold.py
###############################################################################


def test_at():
    """Test paddlecrepe.threshold.At"""
    input_pitch = paddle.to_tensor([100., 110., 120., 130., 140.])
    periodicity = paddle.to_tensor([.19, .22, .25, .17, .30])

    # Perform thresholding
    output_pitch = paddlecrepe.threshold.At(.20)(input_pitch, periodicity)

    # Ensure thresholding is not in-place
    assert not (input_pitch == output_pitch).all()

    # Ensure certain frames are marked as unvoiced
    isnan = paddle.isnan(output_pitch)
    assert isnan[0] and isnan[3]
    assert not isnan[1] and not isnan[2] and not isnan[4]
