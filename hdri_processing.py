"""
HDRI Processing Script.

This script processes HDRI images using the `colour_hdri` library. It applies
various corrections and enhancements, including vignetting, chromatic
aberration, and distortion corrections. The processing is performed using a
graph-based pipeline and supports multiprocessing for efficiency.

Dependencies:
    - colour
    - colour-datasets
    - colour-hdri
    - numpy

Usage:
    Run the script in a Python environment:
    ```bash
    python script_name.py
    ```

Functions:
    weighting_function(a): Applies a double sigmoid function to the input
    value `a`.

Example:
    The script processes all RAW files in the `/Ingestion` directory, applies
    the specified corrections, and outputs the resulting HDRI images in the
    ACEScg colour space.
"""

import logging

import colour
import colour_datasets
import numpy as np

import colour_hdri

colour.utilities.set_default_float_dtype(np.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
)


def weighting_function(a: np.ndarray | float) -> np.ndarray | float:
    """
    Apply a double sigmoid weighting function.

    This function generates weights for HDRI image processing based on the
    specified parameters.

    Args:
        a (numpy.ndarray or float): Input values to be weighted.

    Returns
    -------
    numpy.ndarray or float
        Weighted output values.
    """

    import numpy as np

    eps = np.finfo(float).eps

    try:
        return colour_hdri.generation.double_sigmoid_anchored_function(
            a, 0.025 + eps, 0.2, 0.25, 0.975 - eps
        )
    except RuntimeWarning as e:
        logging.log(f"RuntimeWarning: {e}")
        return np.zeros_like(a)  # Fallback for safety


if __name__ == "__main__":
    """
    Main entry point for the HDRI processing pipeline.

    This section initializes the multiprocessing environment, loads camera
    sensitivities, filters RAW files in the `/Ingestion` directory, and
    executes the HDRI graph processing pipeline.
    """
    import multiprocessing

    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")

    camera_sensitivities = colour.utilities.CanonicalMapping(
        {"Nikon Z 9": colour_datasets.load("8314702")["Estimated"]["Nikon Z 9"]}
    )

    RAW_FILES = colour_hdri.filter_files("/Ingestion", ("NEF",))

    graph = colour_hdri.GraphHDRI()
    graph.set_input("array", RAW_FILES)
    graph.set_input("orientation", "90 CW")
    graph.set_input("batch_size", 6)
    graph.set_input("processes", 2)
    graph.set_input("camera_sensitivities", camera_sensitivities)
    graph.set_input("bypass_exposure_normalisation", True)
    graph.set_input("bypass_preview_image", True)
    graph.set_input("CCT_D_uv", [5333, 0])
    graph.set_input("bypass_watermark", True)
    graph.set_input("output_colourspace", "ACEScg")
    graph.set_input("correct_vignette", True)
    graph.set_input("correct_chromatic_aberration", True)
    graph.set_input("correct_distortion", True)
    graph.set_input(
        "weighting_function",
        weighting_function,
    )
    graph.process()
