"""
Define the unit tests for the :mod:`colour_hdri.calibration.debevec1997`
module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.calibration import (
    camera_response_functions_Debevec1997,
    g_solve,
)
from colour_hdri.exposure import average_luminance
from colour_hdri.sampling import samples_Grossberg2003
from colour_hdri.utilities import ImageStack, filter_files

if TYPE_CHECKING:
    from colour.hints import List

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "ROOT_RESOURCES_CALIBRATION",
    "IMAGES_JPG",
    "TestGSolve",
    "TestCameraResponseFunctionsDebevec1997",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(ROOT_RESOURCES_TESTS, "frobisher_001")

ROOT_RESOURCES_CALIBRATION: str = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_hdri", "calibration"
)

IMAGES_JPG: List[str] = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))


class TestGSolve:
    """
    Define :func:`colour_hdri.calibration.debevec1997.g_solve` definition
    unit tests methods.
    """

    def test_g_solve(self) -> None:
        """Test :func:`colour_hdri.calibration.debevec1997.g_solve` definition."""

        image_stack = ImageStack.from_files(IMAGES_JPG)
        L_l = np.log(
            1
            / average_luminance(
                image_stack.f_number,
                image_stack.exposure_time,
                image_stack.iso,
            )
        )
        samples = samples_Grossberg2003(image_stack.data)

        for i in range(3):
            g, lE = g_solve(samples[..., i], L_l)

            # Lower precision for unit tests under *Github Actions*.
            np.testing.assert_allclose(
                g,
                np.load(
                    os.path.join(ROOT_RESOURCES_CALIBRATION, f"test_g_solve_g_{i}.npy")
                ),
                atol=0.001,
            )

            # Lower precision for unit tests under *Github Actions*.
            np.testing.assert_allclose(
                lE,
                np.load(
                    os.path.join(ROOT_RESOURCES_CALIBRATION, f"test_g_solve_lE_{i}.npy")
                ),
                atol=0.001,
            )


class TestCameraResponseFunctionsDebevec1997:
    """
    Define :func:`colour_hdri.calibration.debevec1997.\
camera_response_functions_Debevec1997` definition unit tests methods.
    """

    def test_camera_response_function_Debevec1997(self) -> None:
        """
        Test :func:`colour_hdri.calibration.debevec1997.\
camera_response_functions_Debevec1997` definition.
        """

        # Lower precision for unit tests under *Github Actions*.
        np.testing.assert_allclose(
            camera_response_functions_Debevec1997(ImageStack.from_files(IMAGES_JPG)),
            np.load(
                os.path.join(
                    ROOT_RESOURCES_CALIBRATION,
                    "test_camera_response_function_Debevec1997_crfs.npy",
                )
            ),
            atol=0.00001,
        )
