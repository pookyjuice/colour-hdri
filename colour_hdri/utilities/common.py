"""
Common Utilities
================

Define the common utilities objects that don't fall in any specific category.
"""

from __future__ import annotations

import os
import re
import typing
from collections import defaultdict

if typing.TYPE_CHECKING:
    from colour.hints import Dict, List, Sequence

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "vivification",
    "vivified_to_dict",
    "path_exists",
    "filter_files",
]


def vivification() -> defaultdict:
    """
    Implement supports for vivification of the underlying dict like
    data-structure, magical!

    Returns
    -------
    :class:`defaultdict`

    Examples
    --------
    >>> vivified = vivification()
    >>> vivified["my"]["attribute"] = 1
    >>> vivified["my"]  # doctest: +SKIP
    defaultdict(<function vivification at 0x...>, {u'attribute': 1})
    >>> vivified["my"]["attribute"]
    1
    """

    return defaultdict(vivification)


def vivified_to_dict(vivified: Dict | defaultdict) -> Dict:
    """
    Convert given vivified data-structure to dictionary.

    Parameters
    ----------
    vivified
        Vivified data-structure.

    Returns
    -------
    :class:`dict`

    Examples
    --------
    >>> vivified = vivification()
    >>> vivified["my"]["attribute"] = 1
    >>> vivified_to_dict(vivified)  # doctest: +SKIP
    {u'my': {u'attribute': 1}}
    """

    if isinstance(vivified, defaultdict):
        vivified = {key: vivified_to_dict(value) for key, value in vivified.items()}
    return vivified


def path_exists(path: str | None) -> bool:
    """
    Return whether given path exists.

    Parameters
    ----------
    path
        Path to check the existence.

    Returns
    -------
    :class:`bool`
        Whether given path exists.

    Examples
    --------
    >>> path_exists(__file__)
    True
    >>> path_exists("")
    False
    """

    if path is None:
        return False
    return os.path.exists(path)


def filter_files(directory: str, extensions: Sequence[str]) -> List[str]:
    """
    Filter given directory for files matching given extensions.

    Parameters
    ----------
    directory
        Directory to filter.
    extensions
        Extensions to filter on.

    Returns
    -------
    :class:`list`
        Filtered files.
    """

    return [
        os.path.join(directory, path)
        for path in filter(
            lambda x: re.search(f"{'|'.join(extensions)}$", x),
            sorted(os.listdir(directory)),
        )
    ]
