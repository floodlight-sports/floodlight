=====================
floodlight.transforms
=====================

Collection of data transformation and processing functions.

.. toctree::
   :maxdepth: 1
   :caption: Submodule Reference

   filter
   spatial
   permutation
   interpolation
   temporal

.. rubric:: Filter

.. currentmodule:: floodlight.transforms.filter
.. autosummary::
   :nosignatures:

   butterworth_lowpass
   savgol_lowpass

.. rubric:: Spatial

.. currentmodule:: floodlight.transforms.spatial
.. autosummary::
   :nosignatures:

   subtract_centroid
   min_max_normalize

.. rubric:: Permutation

.. currentmodule:: floodlight.transforms.permutation
.. autosummary::
   :nosignatures:

   assign_roles

.. rubric:: Interpolation

.. currentmodule:: floodlight.transforms.interpolation
.. autosummary::
   :nosignatures:

   interpolate_linear
   interpolate_polynomial
   interpolate_spline

.. rubric:: Temporal

.. currentmodule:: floodlight.transforms.temporal
.. autosummary::
   :nosignatures:

   resample
