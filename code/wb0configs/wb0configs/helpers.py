import pandas as pd
import os
from pathlib import Path
from typing import Any
import logging
from xml.etree import cElementTree as ET
import pickle
import csv


LOGGER = logging.getLogger("__main__")


def store_xml(file_path: Path, content):
    content.write(file_path)


def load_xml(file_path: Path):
    xml_tree = ET.parse(file_path) # create element tree object
    xml_root = xml_tree.getroot() # get root element
    return xml_tree, xml_root


def merge_xml(xml_tree1, xml_tree2):
    
    xml_root_1 = xml_tree1.getroot()
    for page in xml_tree2.findall('page'):
        xml_root_1.append(page)
    xml_tree1._setroot(xml_root_1)
    return xml_tree1



def store_file(file_path: Path, content, *args):

    file_path = Path(file_path)

    if "pkl" in args:
        file_path = Path(file_path).with_suffix('.pkl')
        with open(file_path, 'wb') as handle:
            pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(f"stored file {file_path}")

    if "csv" in args:
        file_path = Path(file_path).with_suffix('.csv')

        with open(file_path, 'w') as f:
            w = csv.writer(f)

            if type(content) == dict:
                w.writerows(content.items())
            elif type(content) == list:
                w.writerows(content)

        LOGGER.info(f"stored file {file_path}")

    if "pkl" not in args and "csv" not in args:
        LOGGER.info(f"did not store file {file_path}")



def load_file(file_path: Path, **kwars) -> Any:

    ## converters and dtypes
    if "converters" in kwars.keys():
        converters = kwars["converters"]
    else:
        converters = None

    if "dtype" in kwars.keys():
        dtype = kwars["dtype"]
    else:
        dtype = None

    file_path = Path(file_path)

    ## add suffix
    if file_path.suffix == "":
        if "pkl" in kwars["ftype"]:
            file_path = Path(file_path).with_suffix('.pkl')

        elif "csv" in kwars["ftype"]:
            file_path = Path(file_path).with_suffix('.csv')

    ## load data based on file_path
    if file_path.suffix == ".pkl":
        if os.path.exists(file_path):
            LOGGER.info(f"loaded file {file_path}")
            return pd.read_pickle(file_path)#.to_dict()
        else:
            LOGGER.info(f"could not load {file_path}")
            return None

    elif file_path.suffix == ".csv":
        if os.path.exists(file_path):
            LOGGER.info(f"loaded file {file_path}")
            # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
            return pd.read_csv(file_path, header=None)
        else:
            LOGGER.info(f"could not load {file_path}")
            return None



def store_df(file_path: Path, df: pd.DataFrame, *args):
    """
    storing dataframes
    :param file_path:
    :param df:
    :param args:
    :return:
    """
    file_path = Path(file_path)

    if "pkl" in args:
        file_path = Path(file_path).with_suffix('.pkl')
        df.to_pickle(file_path)
        LOGGER.info(f"stored file {file_path}")
    if "csv" in args:
        file_path = Path(file_path).with_suffix('.csv')
        if "index" in args:
            index = True
        else:
            index = False
        df.to_csv(file_path, index=index)
        LOGGER.info(f"stored file {file_path}")

    if "pkl" not in args and "csv" not in args:
        LOGGER.info(f"did not store file {file_path}")


def load_df(file_path: Path, **kwars) -> Any:
    """
    loading dataframes
    :param file_path:
    :param kwars:
    :return:
    """

    ## converters and dtypes
    if "converters" in kwars.keys():
        converters = kwars["converters"]
    else:
        converters = None

    if "dtype" in kwars.keys():
        dtype = kwars["dtype"]
    else:
        dtype = None

    file_path = Path(file_path)

    ## add suffix
    if file_path.suffix == "":
        if "pkl" in kwars["ftype"]:
            file_path = Path(file_path).with_suffix('.pkl')

        elif "csv" in kwars["ftype"]:
            file_path = Path(file_path).with_suffix('.csv')

    ## load data based on file_path
    if file_path.suffix == ".pkl":
        if os.path.exists(file_path):
            LOGGER.info(f"loaded file {file_path}")
            return pd.read_pickle(file_path)
        else:
            LOGGER.info(f"could not load {file_path}")
            return None

    elif file_path.suffix == ".csv":
        if os.path.exists(file_path):
            LOGGER.info(f"loaded file {file_path}")
            return pd.read_csv(file_path, converters = converters, dtype = dtype)
        else:
            LOGGER.info(f"could not load {file_path}")
            return None




