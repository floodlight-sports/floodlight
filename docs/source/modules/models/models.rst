=================
floodlight.models
=================

Collection of data models grouped by category. Each submodule contains model classes that provide data analysis methods based on core objects. Inspired by the `scikit-learn API <https://arxiv.org/abs/1309.0238>`_, each model class contains a ``.fit(...)``-method that 'fits' the model to the data.

.. toctree::
   :maxdepth: 1
   :caption: Submodule Reference

   geometry
   kinematics
   kinetics


.. rubric:: All Available Models

.. currentmodule:: floodlight.models
.. autosummary::
   :nosignatures:

   kinematics.DistanceModel
   kinematics.VelocityModel
   kinematics.AccelerationModel
   geometry.CentroidModel
   kinetics.MetabolicPowerModel


For quick reference, the following computations are available after calling the respective model's ``.fit(...)``-method

.. rubric:: Geometry

.. currentmodule:: floodlight.models.geometry
.. autosummary::
   :nosignatures:

   CentroidModel.centroid
   CentroidModel.centroid_distance
   CentroidModel.stretch_index


.. rubric:: Kinematics

.. currentmodule:: floodlight.models.kinematics
.. autosummary::
   :nosignatures:

   DistanceModel.distance_covered
   DistanceModel.cumulative_distance_covered
   VelocityModel.velocity
   AccelerationModel.acceleration


.. rubric:: Kinetics

.. currentmodule:: floodlight.models.kinetics
.. autosummary::
   :nosignatures:

   MetabolicPowerModel.metabolic_power
   MetabolicPowerModel.cumulative_metabolic_power
   MetabolicPowerModel.equivalent_distance
   MetabolicPowerModel.cumulative_equivalent_distance
