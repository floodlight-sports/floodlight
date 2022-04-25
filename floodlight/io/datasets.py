import json
import os
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

from floodlight.io.utils import extract_zip, download_from_url
from floodlight import XY, Pitch, Events, Code
from settings import DATA_DIR


class EIGDDataset:
    """This dataset loads the EIGD-H data from the *A Unified Taxonomy and Multimodal
    Dataset for Events in Invasion Games* paper [1]_.

    Upon instantiation, the class checks if the data already exists in the repository's
    root ``.data``-folder, and will download the files (~120MB) to this folder if not.


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
            'ec7a6a': ['01-04-00', '01-30-00', '01-19-00', '00-53-00', '00-30-00'],
            }

    Examples
    --------
    >>> from floodlight.io.datasets import EIGDDataset

    >>> dataset = EIGDDataset()
    # get one sample
    >>> teamA, teamB, ball = dataset.get(match="48dcd3", segment="00-06-00")
    # get the corresponding pitch
    >>> pitch = dataset.pitch


    References
    ----------
        .. [1] `Biermann, H., Theiner, J., Bassek, M., Raabe, D., Memmert, D., & Ewerth,
            R. (2021, October). A Unified Taxonomy and Multimodal Dataset for Events in
            Invasion Games. In Proceedings of the 4th International Workshop on
            Multimedia Content Analysis in Sports (pp. 1-10).
            <https://dl.acm.org/doi/abs/10.1145/3475722.3482792>`_
    """

    def __init__(self, dataset_path="eigd_dataset"):
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
        self._EIGD_FRAMERATE = 20

        self._data_dir = os.path.join(DATA_DIR, dataset_path)

        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        if not bool(os.listdir(self._data_dir)):
            self._download_and_extract()

    def get(
        self, match: str = "48dcd3", segment: str = "00-06-00"
    ) -> Tuple[XY, XY, XY]:
        """Get one sample from the EIGD dataset.

        Parameters
        ----------
        match : str, optional
            Match identifier, check Notes section for valid arguments.
            Defaults to the first match ("48dcd3").
        segment : str, optional
            Segment identifier, check Notes section for valid arguments.
            Defaults to the first segment ("00-06-00").

        Returns
        -------
        eigd_dataset: Tuple[XY, XY, XY]
            Returns three XY objects of the form (teamA, teamB, ball)
            for the requested sample.
        """
        file_name = os.path.join(
            self._data_dir, f"{match}_{segment}.{self._EIGD_FILE_EXT}"
        )

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f"Could not load file, check class description for valid match "
                f"and segment values ({file_name})."
            )

        with h5py.File(file_name) as h5f:
            pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}
        return (
            XY(xy=pos_dict["team_a"], framerate=self._EIGD_FRAMERATE),
            XY(xy=pos_dict["team_b"], framerate=self._EIGD_FRAMERATE),
            XY(xy=pos_dict["balls"], framerate=self._EIGD_FRAMERATE),
        )

    @property
    def pitch(self) -> Pitch:
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
        """Downloads an archive file into temporary storage and
        extracts the content to the file system.
        """
        file = f"{DATA_DIR}/eigd.zip"
        with open(file, "wb") as binary_file:
            binary_file.write(download_from_url(self._EIGD_HOST_URL))
        extract_zip(file, self._data_dir)
        os.remove(file)


class ToyDataset:
    """This dataset loads a small set of artifical data for a sample football match
    which are stored in the project repository's root ``.data``-folder.

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
    >>> pitch = dataset.pitch

    """

    def __init__(self, dataset_path="toy_dataset"):
        self._TOY_FRAMERATE = 5
        self._TOY_DIRECTIONS = {
            "HT1": {"Home": "rl", "Away": "lr"},
            "HT2": {"Home": "lr", "Away": "rl"},
        }
        self._data_dir = os.path.join(DATA_DIR, dataset_path)

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

    @property
    def pitch(self) -> Pitch:
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
    """This dataset loads the open StatsBomb data that is provided on the `official
    data repository <https://github.com/statsbomb/open-data>`_.

    Due to the size of the full dataset (~5GB), only metadata (~2MB) are downloaded
    to the repository's root ``.data``-folder upon instanteation while the other data
    are only downloaded on demand.

    Notes
    -----
    The dataset contains result, lineup, and event data for a variety of matches from a
    total of eight different competitions (for example every league match ever played by
    Lionel Messi for FC Barcelona). In addition, for some selected matches, StatsBomb360
    data is available which contains the tracked positions of (some) players at the
    respective times of the captured events. As the data is constantly updated, we
    provide an overview over the stats here but refer to the official repository for
    up-to-date information (last modified 25.04.2022).

        competitions = ['Champions League', 'FA Women's Super League', 'FIFA World Cup',
                        'La Liga', 'NWSL' 'Premier League' 'UEFA Euro'
                        'Women's World Cup']

        seasons = {
            'Champions League' : ['1999/2000', '2003/2004', '2004/2005', '2006/2007',
                                  '2008/2009', '2009/2010', '2010/2011', '2011/2012',
                                  '2012/2013', '2013/2014', '2014/2015', '2015/2016',
                                  '2016/2017', '2017/2018', '2018/2019'],
            'FA Women's Super League' : ['2018/2019', '2019/2020', '2020/2021'],
            'FIFA World Cup': ['2018'],
            'La Liga': ['2004/2005', '2005/2006', '2006/2007', '2007/2008', '2008/2009',
                        '2009/2010', '2010/2011', '2011/2012', '2012/2013', '2013/2014',
                        '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019',
                        '2019/2020', '2020/2021']
            'NWSL': ['2018'],
            'Premier League': ['2003/2004'],
            'UEFA Euro': ['2020'],
            'Women's World Cup': ['2019'],
            }

        number_of_matches = {
            Champions League: {
                1999/2000 : 0, 2003/2004 : 1, 2004/2005 : 1, 2006/2007 : 1,
                2008/2009 : 1, 2009/2010 : 1, 2010/2011 : 1,2011/2012 : 1,
                2012/2013 : 1, 2013/2014 : 1,2014/2015 : 1, 2015/2016 : 1,
                2016/2017 : 1, 2017/2018 : 1, 2018/2019 : 1,
                },
            FA Women's Super League: {
                2018/2019 : 108, 2019/2020 : 87, 2020/2021 : 131,
                },
            FIFA World Cup: {
                2018 : 64,
                },
            La Liga: {
                2004/2005: 7, 2005/2006 : 17, 2006/2007 : 26, 2007/2008 : 28,
                2008/2009 : 31, 2009/2010 : 35, 2010/2011 : 33, 2011/2012 : 37,
                2012/2013 : 32, 2013/2014 : 31, 2014/2015 : 38, 2015/2016 : 33,
                2016/2017 : 34, 2017/2018 : 36, 2018/2019 : 34, 2019/2020 : 33,
                2020/2021 : 35,
            NWSL: {
                2018 : 36,
                },
            Premier League: {2003/2004 : 33},
            UEFA Euro: {
                2020 : 51,
                },
            Women's World Cup: {
                2019 : 52,
                },
        }

    Examples
    --------
    >>> from floodlight.io.datasets import StatsBombOpenDataset

    >>> dataset = StatsBombOpenDataset()
    # get one sample
    >>> events = dataset.get(competition="La Liga", season="2020/2021", match_num=35)
    # get the corresponding pitch
    >>> pitch = dataset.pitch
    """

    def __init__(self, dataset_path="statsbomb_dataset"):
        # setup
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
        self._data_dir = os.path.join(DATA_DIR, dataset_path)
        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        if not bool(os.listdir(self._data_dir)):
            self._download_competition_info()

        # create matches directory and check if match info needs to be downloaded
        self._matches_data_dir = os.path.join(
            self._data_dir, self._STATSBOMB_MATCHES_FOLDERNAME
        )
        if not os.path.isdir(self._matches_data_dir):
            os.makedirs(self._matches_data_dir, exist_ok=True)
        if not bool(os.listdir(self._matches_data_dir)):
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

        # create and update links
        self.competition_and_season_matches = {}
        self.links_competition_to_cID = {}
        self.links_season_to_sID = {}
        self.links_match_to_mID = {}
        self.update_data_links_from_files()

    def get(
        self, competition: str = "La Liga", season: str = "2018/2019", match_num=0
    ) -> Tuple[Events, Events, Events, Events]:
        """Get event data from one match of the StatsBomb dataset.

        Parameters
        ----------
        competition : str, optional
            Competition name for which the match is played, check Notes section for
            possible competitions. Defaults to "La Liga".
        season : str, optional
            Season during which the match is played in the format YYYY/YYYY, check Notes
            section for possible competitions. Defaults to "2020/2021".
        match_num
            Match number relating to the available matches in a season. Defaults to the
            first match at index 0.
        Returns
        -------
        eigd_dataset: Tuple[XY, XY, XY]
            Returns four Events objects of the form (home_events_ht1, home_events_ht2,
            away_events_ht1, away_events_ht2)
            for the requested sample.
        """

        # download events
        # events_filepath = os.path.join(
        #     self._events_data_dir,
        #     f"{list(self.links_match_to_mID[competition][season].values())[match_num]}"
        #     f"{self._STATSBOMB_FILE_EXT}",
        # )
        #
        # events_host_url = (
        #     f"{self._STATSBOMB_SCHEMA}://"
        #     f"{self._STATSBOMB_BASE_URL}/"
        #     f"{self._STATSBOMB_EVENTS_FOLDERNAME}/"
        #     # f"{self.links_competition_to_cID[competition]}/"
        #     # f"{self.links_season_to_sID[competition][season]}/"
        #     f"{list(self.links_match_to_mID[competition][season].values())[match_num]}"
        #     f"{self._STATSBOMB_FILE_EXT}"
        # )
        # threesixty_host_url = (
        #     f"{self._STATSBOMB_SCHEMA}://"
        #     f"{self._STATSBOMB_BASE_URL}/"
        #     f"{self._STATSBOMB_THREESIXTY_FOLDERNAME}/"
        #     f"{list(self.links_match_to_mID[competition][season].values())[match_num]}"
        #     f"{self._STATSBOMB_FILE_EXT}"
        # )

        events = None
        return events

    @property
    def pitch(self) -> Pitch:
        """Returns a Pitch object corresponding to the Toy Dataset."""
        return Pitch.from_template("statsbomb")

    def update_data_links_from_files(self):
        """Creates the dictionaries containing data links between competition, season,
        and matches to the respective cIDs, sIDs, and mIDs for all available matches and
        every competition and season in the StatsBomb dataset.
        """
        # updates on competition level
        competition_info = pd.read_json(
            os.path.join(
                self._data_dir,
                self._STATSBOMB_COMPETITIONS_FILENAME + self._STATSBOMB_FILE_EXT,
            ),
        )
        cIDs = competition_info["competition_id"].unique()
        competitions = competition_info["competition_name"].unique()
        self.links_competition_to_cID.update(
            {competition: cIDs[i] for i, competition in enumerate(competitions)}
        )
        self.links_season_to_sID.update(
            {competition: {} for competition in competitions}
        )
        self.links_match_to_mID.update(
            {competition: {} for competition in competitions}
        )
        self.competition_and_season_matches.update(
            {competition: {} for competition in competitions}
        )

        # loop
        for _, single_season in competition_info.iterrows():
            # update on season level
            cID = str(single_season["competition_id"])
            competition = single_season["competition_name"]
            sID = str(single_season["season_id"])
            season = single_season["season_name"]
            self.links_season_to_sID[competition].update({season: sID})

            # update on match level
            with open(
                os.path.join(
                    os.path.join(self._matches_data_dir, cID),
                    sID + self._STATSBOMB_FILE_EXT,
                ),
                "rb",
            ) as matches_file:
                matches_info = json.load(matches_file)
            self.links_match_to_mID[competition][season] = {
                f"{info['home_team']['home_team_name']} "
                f"vs. "
                f"{info['away_team']['away_team_name']}": info["match_id"]
                for info in matches_info
            }
            self.competition_and_season_matches[competition][season] = [
                f"{info['home_team']['home_team_name']} "
                f"vs. "
                f"{info['away_team']['away_team_name']}"
                for info in matches_info
            ]

    def _download_competition_info(self) -> None:
        """Downloads the json file containing competition information into the file
        system.
        """
        competitions_host_url = (
            f"{self._STATSBOMB_SCHEMA}://"
            f"{self._STATSBOMB_BASE_URL}/"
            f"{self._STATSBOMB_COMPETITIONS_FILENAME}"
            f"{self._STATSBOMB_FILE_EXT}"
        )

        # download file with information of all seasons
        competitions_file = os.path.join(
            self._data_dir,
            self._STATSBOMB_COMPETITIONS_FILENAME + self._STATSBOMB_FILE_EXT,
        )
        with open(competitions_file, "wb") as binary_file:
            binary_file.write(download_from_url(competitions_host_url))

    def _download_matches_info(self) -> None:
        """Downloads the json files containing information about available matches from
        all available seasons into the file system.
        """
        competition_info = pd.read_json(
            os.path.join(
                self._data_dir,
                self._STATSBOMB_COMPETITIONS_FILENAME + self._STATSBOMB_FILE_EXT,
            ),
        )
        for _, single_season in competition_info.iterrows():
            cID = str(single_season["competition_id"])
            sID = str(single_season["season_id"])
            if not os.path.exists(
                os.path.join(
                    os.path.join(self._matches_data_dir, cID),
                    sID + self._STATSBOMB_FILE_EXT,
                )
            ):
                season_host_url = (
                    f"{self._STATSBOMB_SCHEMA}://"
                    f"{self._STATSBOMB_BASE_URL}/"
                    f"{self._STATSBOMB_MATCHES_FOLDERNAME}/"
                    f"{cID}/"
                    f"{sID}"
                    f"{self._STATSBOMB_FILE_EXT}"
                )
                competition_data_dir = os.path.join(self._matches_data_dir, cID)
                if not os.path.isdir(competition_data_dir):
                    os.makedirs(competition_data_dir, exist_ok=True)
                season_file = os.path.join(
                    competition_data_dir, sID + self._STATSBOMB_FILE_EXT
                )
                with open(season_file, "wb") as binary_file:
                    binary_file.write(download_from_url(season_host_url))
