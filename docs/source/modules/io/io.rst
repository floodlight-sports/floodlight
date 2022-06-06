=============
floodlight.io
=============

Collection of file parsing functionalities for different data providers as well as loaders for public datasets.

.. toctree::
   :maxdepth: 1
   :caption: Submodule Reference

   datasets
   dfl
   kinexon
   opta
   secondspectrum
   statsbomb
   statsperform
   tracab
   utils


.. rubric:: Datasets

.. currentmodule:: floodlight.io.datasets
.. autosummary::
   :nosignatures:

   EIGDDataset
   StatsBombOpenDataset
   ToyDataset

.. rubric:: DFL

.. currentmodule:: floodlight.io.dfl
.. autosummary::
   :nosignatures:

   read_event_data_xml
   read_position_data_xml
   read_pitch_from_mat_info_xml
   create_links_from_mat_info

.. rubric:: Kinexon

.. currentmodule:: floodlight.io.kinexon
.. autosummary::
   :nosignatures:

   read_kinexon_file
   get_meta_data
   create_links_from_meta_data
   get_column_names_from_csv

.. rubric:: Opta

.. currentmodule:: floodlight.io.opta
.. autosummary::
   :nosignatures:

   read_f24
   get_opta_feedtype

.. rubric:: Second Spectrum

.. currentmodule:: floodlight.io.secondspectrum
.. autosummary::
   :nosignatures:

   read_secspec_files
   create_links_from_metajson

.. rubric:: StatsBomb

.. currentmodule:: floodlight.io.statsbomb
.. autosummary::
   :nosignatures:

   read_open_statsbomb_event_data_json

.. rubric:: StatsPerform

.. currentmodule:: floodlight.io.statsperform
.. autosummary::
   :nosignatures:

   read_event_data_xml
   read_tracking_data_txt
   read_event_data_from_url
   read_tracking_data_from_url
   read_open_event_data_csv
   read_open_tracking_data_csv
   create_links_from_statsperform_tracking_data_txt
   create_links_from_open_tracking_data_csv

.. rubric:: Tracab (ChyronHego)

.. currentmodule:: floodlight.io.tracab
.. autosummary::
   :nosignatures:

   read_tracab_files
