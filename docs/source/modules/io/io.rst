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
   skillcorner
   sportradar
   statsbomb
   statsperform
   tracab
   utils


.. rubric:: catapultApi
   
.. currentmodule:: floodlight.io.catapultApi
.. autosummary::
   :nosignatures:

   create_links_from_meta_data
   get_meta_data
   get_response
   get_response_data_dict_list
   read_position_data_from_activity
   read_position_data_from_dict_list
   read_response
   write_response
   
.. rubric:: Datasets

.. currentmodule:: floodlight.io.datasets
.. autosummary::
   :nosignatures:

   EIGDDataset
   IDSSEDataset
   StatsBombOpenDataset
   ToyDataset

.. rubric:: DFL

.. currentmodule:: floodlight.io.dfl
.. autosummary::
   :nosignatures:

   read_position_data_xml
   read_event_data_xml
   read_pitch_from_mat_info_xml
   read_teamsheets_from_mat_info_xml

.. rubric:: Kinexon

.. currentmodule:: floodlight.io.kinexon
.. autosummary::
   :nosignatures:

   read_position_data_csv
   create_links_from_meta_data
   get_meta_data
   get_column_names_from_csv

.. rubric:: Opta

.. currentmodule:: floodlight.io.opta
.. autosummary::
   :nosignatures:

   read_event_data_xml
   get_opta_feedtype

.. rubric:: Second Spectrum

.. currentmodule:: floodlight.io.secondspectrum
.. autosummary::
   :nosignatures:

   read_position_data_jsonl
   read_event_data_jsonl
   read_teamsheets_from_meta_json

.. rubric:: Skillcorner

.. currentmodule:: floodlight.io.skillcorner
.. autosummary::
   :nosignatures:

   read_position_data_json

.. rubric:: Sportradar

.. currentmodule:: floodlight.io.sportradar
.. autosummary::
   :nosignatures:

   read_event_data_json

.. rubric:: StatsBomb

.. currentmodule:: floodlight.io.statsbomb
.. autosummary::
   :nosignatures:

   read_open_event_data_json
   read_teamsheets_from_open_event_data_json

.. rubric:: StatsPerform

.. currentmodule:: floodlight.io.statsperform
.. autosummary::
   :nosignatures:

   read_position_data_txt
   read_open_position_data_csv
   read_position_data_from_url
   read_event_data_xml
   read_open_event_data_csv
   read_event_data_from_url
   read_teamsheets_from_position_data_txt
   read_teamsheets_from_event_data_xml
   read_teamsheets_from_open_data_csv

.. rubric:: Tracab (ChyronHego)

.. currentmodule:: floodlight.io.tracab
.. autosummary::
   :nosignatures:

   read_position_data_dat
   read_teamsheets_from_dat
   read_teamsheets_from_meta_json
