"""Multimodal ensemble prediction of curve progression in idiopathic scoliosis.

Layers, from the bottom up:

* ``config``        typed configuration, defaulting to the reference settings
* ``data``          the unified dataset CSV, preprocessing, and image loaders
* ``models``        the individual image and clinical models
* ``ensemble``      late fusion of their predicted probabilities
* ``experiments``   the repeated nested cross-validation protocol
* ``final``         the deployable model bundle and its inference path
* ``cli``           one module per command-line entry point

Research use only. This is not a validated medical device.
"""

__version__ = "0.4.0"
