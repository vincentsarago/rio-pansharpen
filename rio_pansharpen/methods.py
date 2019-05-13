"""Rio-pansharpen Methods."""

from __future__ import division

import numpy


def calculateRatio(rgb, pan, weight):
    """Brovey Ratio."""
    return pan / ((rgb[0] + rgb[1] + rgb[2] * weight) / (2 + weight))


def Brovey(rgb, pan, weight, pan_dtype):
    """
    Brovey Method: Each resampled, multispectral pixel is
    multiplied by the ratio of the corresponding
    panchromatic pixel intensity to the sum of all the
    multispectral intensities.
    """
    with numpy.errstate(invalid="ignore", divide="ignore"):
        ratio = calculateRatio(rgb, pan, weight)

    with numpy.errstate(invalid="ignore"):
        sharp = numpy.clip(ratio * rgb, 0, numpy.iinfo(pan_dtype).max)
        return sharp.astype(pan_dtype), ratio
