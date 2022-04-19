import os
import tempfile
from typing import Tuple

import h5py

from floodlight.io.utils import extract_zip, download_from_url
from floodlight import XY, Pitch
from settings import DATA_DIR


class EIGDDataset:
    """
    Notes
    -----
    matches = ['48dcd3', 'ad969d', 'e0e547', 'e8a35a', 'ec7a6a']
    segments = {
        '48dcd3': ['00-06-00', '00-15-00', '00-25-00', '01-05-00', '01-10-00'],
        'ad969d': ['00-00-30', '00-15-00', '00-43-00', '01-11-00', '01-35-00'],
        'e0e547': ['00-00-00', '00-08-00', '00-15-00', '00-50-00', '01-00-00'],
        'e8a35a': ['00-02-00', '00-07-00', '00-14-00', '01-05-00', '01-14-00'],
        'ec7a6a': ['01-04-00', '01-30-00', '01-19-00', '00-53-00', '00-30-00'],
        }
    """

    def __init__(self, dataset_path="eigd_dataset"):
        self._EIGD_SCHEMA = "https"
        self._EIGD_BASE_URL = "data.uni-hannover.de/dataset/8ccb364e-145f-4b28-8ff4-954b86e9b30d/resource/fd24e032-742d-4609-9052-cec310a2a563/download"
        self._EIGD_FILENAME = "eigd-h_pos.zip"
        self._EIGD_HOST_URL = (
            f"{self._EIGD_SCHEMA}://{self._EIGD_BASE_URL}/{self._EIGD_FILENAME}"
        )
        self._EIGD_FILE_EXT = "h5"
        self._EIGD_FRAMERATE = 20

        self._data_dir = os.path.join(DATA_DIR, dataset_path)

        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        if not bool(os.listdir(self._data_dir)):
            self._download_and_extract()

    def get(
        self, match: str = "48dcd3", segment: str = "00-06-00"
    ) -> Tuple[XY, XY, XY]:
        """

        Parameters
        ----------
        match
        segment

        Returns
        -------

        """
        file_name = os.path.join(
            self._data_dir, f"{match}_{segment}.{self._EIGD_FILE_EXT}"
        )

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f"Could not load file, check class description for valid match and segment values ({file_name})."
            )

        with h5py.File(file_name) as h5f:
            pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}
        return (
            XY(xy=pos_dict["team_a"], framerate=self._EIGD_FRAMERATE),
            XY(xy=pos_dict["team_b"], framerate=self._EIGD_FRAMERATE),
            XY(xy=pos_dict["balls"], framerate=self._EIGD_FRAMERATE),
        )

    @property
    def get_pitch(self) -> Pitch:
        """Returns a Pitch object corresponding to the EIGD-data."""
        return Pitch(
            xlim=(0, 40),
            ylim=(0, 20),
            unit="m",
            boundaries="fixed",
            length=40,
            width=20,
            sport="handball",
        )

    def _download_and_extract(self) -> None:
        """downloads an archive file into temporary storage and extracts the content to the file system."""
        tmp = tempfile.NamedTemporaryFile()
        tmp.write(download_from_url(self._EIGD_HOST_URL))
        extract_zip(tmp.name, self._data_dir)
        tmp.close()
