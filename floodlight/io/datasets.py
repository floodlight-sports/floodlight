import json
import os
from typing import Tuple
from urllib.error import HTTPError, URLError

import h5py
import numpy as np
import pandas as pd

from floodlight.io.utils import extract_zip, download_from_url
from floodlight.io.statsbomb import read_open_statsbomb_event_data_json
from floodlight import XY, Pitch, Events, Code
from floodlight.settings import DATA_DIR


class EIGDDataset:
    """This dataset loads the EIGD-H data from the *A Unified Taxonomy and Multimodal
    Dataset for Events in Invasion Games* paper. [1]_

    Upon instantiation, the class checks if the data already exists in the repository's
    root ``.data``-folder, and will download the files (~120MB) to this folder if not.

    Parameters
    ----------
    dataset_dir_name: str, optional
        Name of subdirectory where the dataset is stored within the root .data
        directory. Defaults to 'eigd_dataset'.

    Notes
    -----
    The dataset contains a total of 25 short samples of spatiotemporal data for both
    teams and the ball from the German Men's Handball Bundesliga (HBL). For more
    information, visit the
    `official project repository <https://github.com/MM4SPA/eigd>`_.
    Data for one sample can be queried calling the :func:`~EIGDDataset.get`-method
    specifying the match and segment. The following matches and segments are
    available::

        matches = ['48dcd3', 'ad969d', 'e0e547', 'e8a35a', 'ec7a6a']
        segments = {
            '48dcd3': ['00-06-00', '00-15-00', '00-25-00', '01-05-00', '01-10-00'],
            'ad969d': ['00-00-30', '00-15-00', '00-43-00', '01-11-00', '01-35-00'],
            'e0e547': ['00-00-00', '00-08-00', '00-15-00', '00-50-00', '01-00-00'],
            'e8a35a': ['00-02-00', '00-07-00', '00-14-00', '01-05-00', '01-14-00'],
            'ec7a6a': ['00-30-00', '00-53-00', '01-19-00', '01-30-00', '01-40-00'],
            }

    Examples
    --------
    >>> from floodlight.io.datasets import EIGDDataset

    >>> dataset = EIGDDataset()
    # get one sample
    >>> teamA, teamB, ball = dataset.get(match_name="48dcd3", segment="00-06-00")
    # get the corresponding pitch
    >>> pitch = dataset.get_pitch()


    References
    ----------
        .. [1] `Biermann, H., Theiner, J., Bassek, M., Raabe, D., Memmert, D., & Ewerth,
            R. (2021, October). A Unified Taxonomy and Multimodal Dataset for Events in
            Invasion Games. In Proceedings of the 4th International Workshop on
            Multimedia Content Analysis in Sports (pp. 1-10).
            <https://dl.acm.org/doi/abs/10.1145/3475722.3482792>`_
    """

    def __init__(self, dataset_dir_name="eigd_dataset"):
        self._EIGD_SCHEMA = "https"
        self._EIGD_BASE_URL = (
            "data.uni-hannover.de/dataset/8ccb364e-145f-4b28-8ff4-954b86e9b30d/"
            "resource/fd24e032-742d-4609-9052-cec310a2a563/download"
        )
        self._EIGD_FILENAME = "eigd-h_pos.zip"
        self._EIGD_HOST_URL = (
            f"{self._EIGD_SCHEMA}://{self._EIGD_BASE_URL}/{self._EIGD_FILENAME}"
        )
        self._EIGD_FILE_EXT = "h5"
        self._EIGD_FRAMERATE = 30

        self._data_dir = os.path.join(DATA_DIR, dataset_dir_name)

        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        if not bool(os.listdir(self._data_dir)):
            self._download_and_extract()

    def get(
        self, match_name: str = "48dcd3", segment: str = "00-06-00"
    ) -> Tuple[XY, XY, XY]:
        """Get one sample from the EIGD dataset.

        Parameters
        ----------
        match_name : str, optional
            Match name, check Notes section for valid arguments.
            Defaults to the first match ("48dcd3").
        segment : str, optional
            Segment identifier, check Notes section for valid arguments.
            Defaults to the first segment ("00-06-00").

        Returns
        -------
        sample: Tuple[XY, XY, XY]
            Returns three XY objects of the form (teamA, teamB, ball)
            for the requested sample.
        """
        file_name = os.path.join(
            self._data_dir, f"{match_name}_{segment}.{self._EIGD_FILE_EXT}"
        )

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f"Could not load file, check class description for valid match "
                f"and segment values ({file_name})."
            )

        # extract from file
        with h5py.File(file_name) as h5f:
            pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}

        # assemble
        sample = (
            XY(xy=self._transform(pos_dict["team_a"]), framerate=self._EIGD_FRAMERATE),
            XY(xy=self._transform(pos_dict["team_b"]), framerate=self._EIGD_FRAMERATE),
            XY(xy=self._transform(pos_dict["balls"]), framerate=self._EIGD_FRAMERATE),
        )

        return sample

    @staticmethod
    def get_pitch() -> Pitch:
        """Returns a Pitch object corresponding to the EIGD-data."""
        return Pitch.from_template("eigd")

    @staticmethod
    def _transform(data: np.ndarray) -> np.ndarray:
        """Transforms spatiotemporal data from EIGD-format to floodlight format.

        Parameters
        ----------
        data: np.ndarray
            Array of shape (T, N, xydim), with T time dimension, N the number of players
            and xydim a separate dimension for x-, y-, and z-coordinates (ball only).

        Returns
        -------
        data_transformed: np.ndarray
            Array of shape (T, N*2), with T time dimension and N the number of players.
            All z-coordinates are omitted to match typical floodlight format.
        """
        # EIDG data is stored in 3-dimensional array, extract size and reshape
        T, N, _ = data.shape
        data_transformed = data[:, :, :2].reshape((T, N * 2))

        return data_transformed

    def _download_and_extract(self) -> None:
        """Downloads an archive file into temporary storage and
        extracts the content to the file system.
        """
        file = f"{DATA_DIR}/eigd.zip"
        with open(file, "wb") as binary_file:
            binary_file.write(download_from_url(self._EIGD_HOST_URL))
        extract_zip(file, self._data_dir)
        os.remove(file)


class ToyDataset:
    """This dataset loads synthetic data for a (very) short artificial football match.

    The data can be used for testing or trying out features. They come shipped with the
    package and are stored in the repository's root ``.data``-folder.

    Examples
    --------
    >>> from floodlight.io.datasets import ToyDataset

    >>> dataset = ToyDataset()
    # get one sample
    >>> (
    >>>     xy_home,
    >>>     xy_away,
    >>>     xy_ball,
    >>>     events_home,
    >>>     events_away,
    >>>     possession,
    >>>     ballstatus,
    >>> ) = dataset.get(segment="HT1")
    # get the corresponding pitch
    >>> pitch = dataset.get_pitch()

    """

    def __init__(self):
        self._TOY_FRAMERATE = 5
        self._TOY_DIRECTIONS = {
            "HT1": {"Home": "rl", "Away": "lr"},
            "HT2": {"Home": "lr", "Away": "rl"},
        }
        self._data_dir = os.path.join(DATA_DIR, "toy_dataset")

    def get(
        self, segment: str = "HT1"
    ) -> Tuple[XY, XY, XY, Events, Events, Code, Code]:
        """Get data objects for one segment from the toy dataset.

        Parameters
        ----------
        segment : str, optional
            Segment identifier for the first ("HT1") or the second ("HT2") half.
            Defaults to the first half ("HT1").

        Returns
        -------
        toy_dataset:  Tuple[XY, XY, XY, Events, Events, Code, Code]
            Returns seven XY objects of the form (xy_home, xy_away, xy_ball,
            events_home, events_away, possession, ballstatus) for the requested segment.
        """

        if segment not in ["HT1", "HT2"]:
            raise FileNotFoundError(
                f"Expected segment to be of 'HT1' or 'HT2', got {segment}"
            )

        xy_home = XY(
            xy=np.load(os.path.join(self._data_dir, f"xy_home_{segment.lower()}.npy")),
            framerate=self._TOY_FRAMERATE,
            direction=self._TOY_DIRECTIONS[segment]["Home"],
        )

        xy_away = XY(
            xy=np.load(os.path.join(self._data_dir, f"xy_away_{segment.lower()}.npy")),
            framerate=self._TOY_FRAMERATE,
            direction=self._TOY_DIRECTIONS[segment]["Away"],
        )

        xy_ball = XY(
            xy=np.load(os.path.join(self._data_dir, f"xy_ball_{segment.lower()}.npy")),
            framerate=self._TOY_FRAMERATE,
        )

        events_home = Events(
            events=pd.read_csv(
                os.path.join(self._data_dir, f"events_home_{segment.lower()}.csv")
            )
        )

        events_away = Events(
            events=pd.read_csv(
                os.path.join(self._data_dir, f"events_away_{segment.lower()}.csv")
            )
        )

        possession = Code(
            code=np.load(
                os.path.join(self._data_dir, f"possession_{segment.lower()}.npy")
            ),
            name="possession",
            definitions={1: "Home", 2: "Away"},
            framerate=self._TOY_FRAMERATE,
        )

        ballstatus = Code(
            code=np.load(
                os.path.join(self._data_dir, f"ballstatus_{segment.lower()}.npy")
            ),
            name="ballstatus",
            definitions={0: "Dead", 1: "Alive"},
            framerate=self._TOY_FRAMERATE,
        )

        data_objects = (
            xy_home,
            xy_away,
            xy_ball,
            events_home,
            events_away,
            possession,
            ballstatus,
        )

        return data_objects

    @staticmethod
    def get_pitch() -> Pitch:
        """Returns a Pitch object corresponding to the Toy Dataset."""
        return Pitch(
            xlim=(-52.5, 52.5),
            ylim=(-34, 34),
            unit="m",
            boundaries="flexible",
            length=105,
            width=68,
            sport="football",
        )


class StatsBombOpenDataset:
    """This dataset loads the StatsBomb open data provided by the `official
    data repository <https://github.com/statsbomb/open-data>`_.

    Due to the size of the full dataset (~5GB), only metadata (~2MB) are downloaded
    to the repository's root ``.data``-folder upon instantiation while the other data
    are only downloaded on demand. All downloaded files stay on disk if not manually
    removed.

    Parameters
    ----------
    dataset_dir_name: str, optional
        Name of subdirectory where the dataset is stored within the root .data
        directory. Defaults to 'statsbomb_dataset'.

    Notes
    -----
    The dataset contains results, lineups, event data, and (partly) `StatsBomb360 data
    <https://statsbomb.com/articles/soccer/
    statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/>`_ for a variety
    of matches from a total of eight different competitions (Women's World Cup,
    FIFA World Cup, UEFA Euro, Champions League, FA Women's Super League, NWSL,
    Premier League, and La Liga).
    The Champions League data for example contains all Finals from 2003/2004 to
    2018/2019.
    The La Liga data contains every one of the 520 matches ever played by Lionel Messi
    for FC Barcelona.
    The UEFA Euro data contains 51 matches where StatsBomb360 data is available.
    As the data is constantly updated, we provide an overview over the stats here but
    refer to the official repository for up-to-date information (last
    checked 20.08.2022)::

        number_of_matches = {
            "Champions League": {
                '1999/2000' : 0, '2003/2004' : 1, '2004/2005' : 1, '2006/2007' : 1,
                '2008/2009' : 1, '2009/2010' : 1, '2010/2011' : 1, '2011/2012' : 1,
                '2012/2013' : 1, '2013/2014' : 1, '2014/2015' : 1, '2015/2016' : 1,
                '2016/2017' : 1, '2017/2018' : 1, '2018/2019' : 1,
                },
            "FA Women's Super League": {
                '2018/2019' : 108, '2019/2020' : 87, '2020/2021' : 131,
                },
            "FIFA World Cup": {
                '2018' : 64,
                },
            "La Liga": {
                '2004/2005': 7, '2005/2006' : 17, '2006/2007' : 26, '2007/2008' : 28,
                '2008/2009' : 31, '2009/2010' : 35, '2010/2011' : 33, '2011/2012' : 37,
                '2012/2013' : 32, '2013/2014' : 31, '2014/2015' : 38, '2015/2016' : 33,
                '2016/2017' : 34, '2017/2018' : 36, '2018/2019' : 34, '2019/2020' : 33,
                '2020/2021' : 35,
                },
            "NWSL": {
                '2018' : 36,
                },
            "Premier League": {
                '2003/2004' : 33,
                },
            "UEFA Euro" : {
                '2020' : 51,
                },
            "Women's World Cup": {
                '2019' : 52,
                },
        }

    Examples
    --------
    >>> from floodlight.io.datasets import StatsBombOpenDataset
    >>> dataset = StatsBombOpenDataset()
    # get one sample of event data with StatsBomb360 data
    >>> events = dataset.get("UEFA Euro", "2020", "England vs. Germany")
    # get the corresponding pitch
    >>> pitch = dataset.get_pitch()
    # get a summary of available matches in the dataset
    >>> matches = dataset.available_matches
    # extract every La Liga Clásico played in Camp Nou by Lionel Messi
    >>> clasicos = matches[matches["match_name"] == "Barcelona vs. Real Madrid"]
    # print outcomes
    >>> for _, match in clasicos.iterrows():
    >>>     print(f"Season {match['season_name']} - Barcelona {match['score']} Real'")
    # read events to list
    >>> clasico_events = []
    >>> for _, clasico in clasicos.iterrows():
    >>>     data = dataset.get("La Liga", clasico["season_name"], clasico["match_name"])
    >>>     clasico_events.append(data)
    """

    def __init__(self, dataset_dir_name="statsbomb_dataset"):
        # setup
        self._links_competition_to_cID = {}
        self._links_season_to_sID = {}
        self._links_match_to_mID = {}
        self._STATSBOMB_SCHEMA = "https"
        self._STATSBOMB_BASE_URL = (
            "raw.githubusercontent.com/statsbomb/open-data/master/data"
        )
        self._STATSBOMB_COMPETITIONS_FILENAME = "competitions"
        self._STATSBOMB_MATCHES_FOLDERNAME = "matches"
        self._STATSBOMB_EVENTS_FOLDERNAME = "events"
        self._STATSBOMB_THREESIXTY_FOLDERNAME = "three-sixty"
        self._STATSBOMB_FILE_EXT = ".json"

        # create data directory and check if competition info needs to be downloaded
        self._data_dir = os.path.join(DATA_DIR, dataset_dir_name)
        self.filepath_competitions = os.path.join(
            self._data_dir,
            self._STATSBOMB_COMPETITIONS_FILENAME + self._STATSBOMB_FILE_EXT,
        )
        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        if not os.path.exists(self.filepath_competitions):
            self._download_competition_info()

        # create matches directory and check if match info needs to be downloaded
        self._matches_data_dir = os.path.join(
            self._data_dir, self._STATSBOMB_MATCHES_FOLDERNAME
        )
        if not os.path.isdir(self._matches_data_dir):
            os.makedirs(self._matches_data_dir, exist_ok=True)
        self._download_matches_info()

        # create events location
        self._events_data_dir = os.path.join(
            self._data_dir, self._STATSBOMB_EVENTS_FOLDERNAME
        )
        if not os.path.isdir(self._events_data_dir):
            os.makedirs(self._events_data_dir, exist_ok=True)

        # create StatsBomb360 location
        self._threesixty_data_dir = os.path.join(
            self._data_dir, self._STATSBOMB_THREESIXTY_FOLDERNAME
        )
        if not os.path.isdir(self._threesixty_data_dir):
            os.makedirs(self._threesixty_data_dir, exist_ok=True)

        # read links from files and update class level dictionaries
        self._read_competition_links_from_file()

    @property
    def available_matches(self) -> pd.DataFrame:
        """Creates and returns a DataFrame with information for all available matches
        from the metadata that is downloaded upon instantiation.

        Returns
        -------
        summary: pd.DataFrame
            Table where the rows contain meta information of individual games such as
            ``competition_name``, ``season_name``, and ``match_name`` (in the format
            Home vs. Away), location of the match (``stadium`` and ``country``),
            ``sex`` of the players (female or male), the ``StatsBomb360_status``  and
            the final ``score``.
        """
        summary = pd.DataFrame()

        # loop over season and competition
        for competition in self._links_competition_to_cID:
            cID = self._links_competition_to_cID[competition]
            self._read_season_match_links_for_competition_from_files(competition)
            for season in self._links_season_to_sID[competition]:
                sID = self._links_season_to_sID[competition][season]

                # loop over matches
                filepath_matches = os.path.join(
                    os.path.join(self._matches_data_dir, str(cID)),
                    str(sID) + self._STATSBOMB_FILE_EXT,
                )
                with open(filepath_matches, "r", encoding="utf8") as f:
                    matchinfo_list = json.load(f)

                for info in matchinfo_list:
                    match_info = {
                        "competition_name": competition,
                        "season_name": season,
                        "match_name": f"{info['home_team']['home_team_name']} "
                        f"vs. "
                        f"{info['away_team']['away_team_name']}",
                        "score": f"{info['home_score']}:{info['away_score']}",
                        "stadium": info["stadium"]["name"],
                        "country": info["stadium"]["country"]["name"],
                        "sex": "f"
                        if competition
                        in ["FA Women's Super League", "NWSL", "Women's World Cup"]
                        else "m",
                        "StatsBomb360_status": info["match_status_360"],
                        "cID": cID,
                        "sID": sID,
                        "mID": info["match_id"],
                    }
                    summary = summary.append(match_info, ignore_index=True)
        return summary

    def get(
        self,
        competition_name: str = "La Liga",
        season_name: str = "2020/2021",
        match_name: str = None,
    ) -> Tuple[Events, Events, Events, Events]:
        """Get events from one match of the StatsBomb open dataset.

        If `StatsBomb360data <https://statsbomb.com/articles/soccer/
        statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/>`_  are
        available, they are stored in the  ``qualifier`` column of the Events object.
        If the files are not contained in the repository's root ``.data`` folder they
        are downloaded to the folder and will be stored until removed by hand.

        Parameters
        ----------
        competition_name : str, optional
            Competition name for which the match is played, check Notes section for
            possible competitions. Defaults to "La Liga".
        season_name : str, optional
            Season name during which the match is played. For league matches use the
            format YYYY/YYYY and for international cup matches the format YYYY.
            Check Notes for available seasons of every competition.
            Defaults to "2020/2021".
        match_name: str, optional
            Match name relating to the available matches in the chosen competition and
            season. If equal to None (default), the first available match of the
            given competition and season is chosen.

        Returns
        -------
        data_objects: Tuple[Events, Events, Events, Events]
            Returns four Events objects of the form (events_home_ht1, events_home_ht2,
            events_away_ht1, events_away_ht2) for the requested sample.
        """
        # get identifiers from links
        cID = self._links_competition_to_cID[competition_name]
        if competition_name not in self._links_season_to_sID:
            self._read_season_match_links_for_competition_from_files(competition_name)
        sID = self._links_season_to_sID[competition_name][season_name]
        matches_dict = self._links_match_to_mID[competition_name][season_name]
        if match_name is None:
            mID = list(matches_dict.values())[0]
        else:
            mID = matches_dict[match_name]

        # create paths
        filepath_matches = os.path.join(
            os.path.join(self._matches_data_dir, str(cID)),
            str(sID) + self._STATSBOMB_FILE_EXT,
        )
        filepath_events = os.path.join(
            self._events_data_dir,
            str(mID) + self._STATSBOMB_FILE_EXT,
        )
        filepath_threesixty = os.path.join(
            self._threesixty_data_dir,
            str(mID) + self._STATSBOMB_FILE_EXT,
        )

        # check if events need to be downloaded
        if not os.path.exists(filepath_events):
            events_host_url = (
                f"{self._STATSBOMB_SCHEMA}://"
                f"{self._STATSBOMB_BASE_URL}/"
                f"{self._STATSBOMB_EVENTS_FOLDERNAME}/"
                f"{str(mID)}"
                f"{self._STATSBOMB_FILE_EXT}"
            )
            with open(filepath_events, "wb") as binary_file:
                binary_file.write(download_from_url(events_host_url))

        # check if StatsBomb360 data is available and needs to be downloaded
        if not os.path.exists(filepath_threesixty):
            threesixty_host_url = (
                f"{self._STATSBOMB_SCHEMA}://"
                f"{self._STATSBOMB_BASE_URL}/"
                f"{self._STATSBOMB_THREESIXTY_FOLDERNAME}/"
                f"{str(mID)}"
                f"{self._STATSBOMB_FILE_EXT}"
            )
            try:
                data = download_from_url(threesixty_host_url)
                with open(filepath_threesixty, "wb") as binary_file:
                    binary_file.write(data)
            except HTTPError:
                filepath_threesixty = None

        # read events from file
        (home_ht1, home_ht2, away_ht1, away_ht2,) = read_open_statsbomb_event_data_json(
            filepath_events, filepath_matches, filepath_threesixty
        )
        event_objects = (home_ht1, home_ht2, away_ht1, away_ht2)

        return event_objects

    @staticmethod
    def get_pitch() -> Pitch:
        """Returns a Pitch object corresponding to the StatsBomb Dataset."""
        return Pitch.from_template("statsbomb", sport="football")

    def _read_competition_links_from_file(self):
        """Writes the data links between the available competitions and the respective
        cIDs to the class level dictionary.
        """
        # updates on competition level
        competition_info = pd.read_json(self.filepath_competitions)
        cIDs = competition_info["competition_id"].unique()
        competitions = competition_info["competition_name"].unique()
        self._links_competition_to_cID.update(
            {competition: cIDs[i] for i, competition in enumerate(competitions)}
        )

    def _read_season_match_links_for_competition_from_files(self, competition_name):
        """Writes data links between the seasons and matches to the respective sIDs
        and mIDs for a given competition to the class level dictionaries.
        """
        # read competition file
        cID = self._links_competition_to_cID[competition_name]
        competition_info = pd.read_json(self.filepath_competitions)

        # update season and match dictionaries with competition information
        self._links_season_to_sID.update({competition_name: {}})
        self._links_match_to_mID.update({competition_name: {}})

        # loop over all available seasons of the given competition
        for _, single_season in competition_info.iterrows():
            if cID != single_season["competition_id"]:
                continue

            # update season and match dictionaries with season information
            sID = single_season["season_id"]
            season_name = single_season["season_name"]
            self._links_season_to_sID[competition_name].update({season_name: sID})
            self._links_match_to_mID[competition_name].update({season_name: {}})

            # read information of all matches within the season
            filepath_matches = os.path.join(
                os.path.join(self._matches_data_dir, str(cID)),
                str(sID) + self._STATSBOMB_FILE_EXT,
            )
            with open(filepath_matches, "rb") as matches_file:
                season_matches_info = json.load(matches_file)

            # update match dictionary with match information
            for info in season_matches_info:
                match_name = (
                    f"{info['home_team']['home_team_name']} vs. "
                    f"{info['away_team']['away_team_name']}"
                )
                mID = info["match_id"]
                self._links_match_to_mID[competition_name][season_name].update(
                    {match_name: mID}
                )

    def _download_competition_info(self) -> None:
        """Downloads json file with competition information into the file system."""
        competitions_host_url = (
            f"{self._STATSBOMB_SCHEMA}://"
            f"{self._STATSBOMB_BASE_URL}/"
            f"{self._STATSBOMB_COMPETITIONS_FILENAME}"
            f"{self._STATSBOMB_FILE_EXT}"
        )

        # download file with information of all seasons
        try:
            with open(self.filepath_competitions, "wb") as binary_file:
                binary_file.write(download_from_url(competitions_host_url))
        except URLError:  # remove empty json file if download fails
            os.remove(self.filepath_competitions)
            raise URLError(
                f"Could not download competitions.json from URL "
                f"{competitions_host_url}. Check your internet connection!"
            )

    def _download_matches_info(self) -> None:
        """Downloads the json files containing information about available matches from
        all available seasons into the file system.
        """
        competition_info = pd.read_json(self.filepath_competitions)

        for _, single_season in competition_info.iterrows():
            cID = single_season["competition_id"]
            sID = single_season["season_id"]
            matches_filepath = os.path.join(
                os.path.join(self._matches_data_dir, str(cID)),
                str(sID) + self._STATSBOMB_FILE_EXT,
            )
            if not os.path.exists(matches_filepath):
                season_host_url = (
                    f"{self._STATSBOMB_SCHEMA}://"
                    f"{self._STATSBOMB_BASE_URL}/"
                    f"{self._STATSBOMB_MATCHES_FOLDERNAME}/"
                    f"{str(cID)}/"
                    f"{str(sID)}"
                    f"{self._STATSBOMB_FILE_EXT}"
                )
                competition_data_dir = os.path.join(self._matches_data_dir, str(cID))
                if not os.path.isdir(competition_data_dir):
                    os.makedirs(competition_data_dir, exist_ok=True)
                season_file = os.path.join(
                    competition_data_dir, str(sID) + self._STATSBOMB_FILE_EXT
                )
                with open(season_file, "wb") as binary_file:
                    binary_file.write(download_from_url(season_host_url))
