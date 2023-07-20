import numpy as np
import pandas as pd

from src.db.zip_code_mapper import ZipCodeMapper


def pos_from_coords(coords: pd.DataFrame) -> dict:
    """
    Computes a layout dictionary of the form
        {node1: [long1, lat1], node2: [long2, lat2], ...}
    that can be used for plotting (required format by networkx).
    :param coords: DataFrame of coordinates with columns "lat" and "long" and the node ids as index
    :return: dict of node positions
    """
    # compute graph layout
    # use geographical coordinates of customers for graph layout
    fixed_zip_pos = coords.to_dict("index")
    # map dict of dicts to dict of tuples
    fixed_zip_pos = {key: (values["long"], values["lat"]) for key, values in fixed_zip_pos.items()}
    return fixed_zip_pos


def circular_layout_around_zip_codes(fixed_zip_pos: dict, radius: float = 0.2) -> dict:
    """
    Takes a dictionary of node positions and distributes the nodes randomly
    within a circle around their original positions.
    :param fixed_zip_pos: node layout. Dictionary of list-like coordinates.
    :param radius: disc radius
    :return: dictionary of new node positions
    """
    pos = dict()
    for node, coord in fixed_zip_pos.items():
        x, y = coord
        rand_x, rand_y = np.random.rand(2)
        pos[node] = [x + radius*np.cos(rand_x*2*np.pi), y + radius*np.sin(rand_y*2*np.pi)]
    return pos


def geo_layout(adr_zips) -> dict:
    """
    Maps a list of zip codes to geographical coordinates. The result will be a layout dictionary
    of the form
        {node1: [long1, lat1], node2: [long2, lat2], ...}
    To avoid mapping repeated zip codes to the same location, we scatter the points randomly in a circle
    around each location.
    :param adr_zips: list-like of zip codes
    :return: dictionary of node positions
    """
    # load mapper for zip_code -> (longitude, latitude)
    zip_mapper = ZipCodeMapper()
    # map zip code to coords
    coords = zip_mapper.map_zip_codes_to_coords(adr_zips)
    # remove zip codes, keep lat and long
    coords.drop(columns="adr_zip", inplace=True)
    # get dict of node positions
    fixed_zip_pos = pos_from_coords(coords)
    # scatter customers in a circle around their zip code
    pos = circular_layout_around_zip_codes(fixed_zip_pos, radius=0.1)
    return pos


top_cities = {
    'Berlin': (13.404954, 52.520008),
    'Cologne': (6.953101, 50.935173),
    'Dortmund': (7.468554, 51.513400),
    'Düsseldorf': (6.782048, 51.227144),
    'Frankfurt am Main': (8.682127, 50.110924),
    'Hamburg': (9.993682, 53.551086),
    'Hannover': (9.73322, 52.37052),
    'Leipzig': (12.387772, 51.343479),
    'Munich': (11.576124, 48.137154),
    'Nuremberg': (11.077438, 49.449820),
    'Stuttgart': (9.181332, 48.777128)
}
