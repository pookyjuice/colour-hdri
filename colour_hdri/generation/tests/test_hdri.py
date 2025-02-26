"""Define the unit tests for the :mod:`colour_hdri.generation.radiance` module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from colour import RGB_COLOURSPACES

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.calibration import camera_response_functions_Debevec1997
from colour_hdri.generation import image_stack_to_HDRI
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
    "ROOT_RESOURCES_GENERATION",
    "IMAGES_JPG",
    "TestImageStackToHDRI",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(ROOT_RESOURCES_TESTS, "frobisher_001")

ROOT_RESOURCES_GENERATION: str = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_hdri", "generation"
)

IMAGES_JPG: List[str] = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))


class TestImageStackToHDRI:
    """
    Define :func:`colour_hdri.generation.radiance.image_stack_to_HDRI`
    definition unit tests methods.
    """

    def test_image_stack_to_HDRI(self) -> None:
        """
        Test :func:`colour_hdri.generation.radiance.image_stack_to_HDRI`
        definition.
        """

        image_stack = ImageStack.from_files(IMAGES_JPG)
        image_stack.data = RGB_COLOURSPACES["sRGB"].cctf_decoding(image_stack.data)

        np.testing.assert_allclose(
            image_stack_to_HDRI(image_stack),
            np.load(
                os.path.join(
                    ROOT_RESOURCES_GENERATION,
                    "test_image_stack_to_hdri_linear.npy",
                )
            ),
            atol=0.0001,
        )

        image_stack = ImageStack.from_files(IMAGES_JPG)
        np.testing.assert_allclose(
            image_stack_to_HDRI(
                image_stack,
                camera_response_functions=(
                    camera_response_functions_Debevec1997(image_stack)
                ),
            ),
            np.load(
                os.path.join(
                    ROOT_RESOURCES_GENERATION,
                    "test_image_stack_to_hdri_crfs.npy",
                )
            ),
            atol=0.0001,
        )

        image_stack = ImageStack.from_files(IMAGES_JPG, read_data=False)
        np.testing.assert_allclose(
            image_stack_to_HDRI(
                image_stack,
                camera_response_functions=(
                    camera_response_functions_Debevec1997(image_stack)
                ),
            ),
            np.load(
                os.path.join(
                    ROOT_RESOURCES_GENERATION,
                    "test_image_stack_to_hdri_crfs.npy",
                )
            ),
            atol=0.0001,
        )
