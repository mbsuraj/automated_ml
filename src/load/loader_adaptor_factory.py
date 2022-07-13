import os
from src.load.csv_loader import CSVLoader

class LoaderAdaptorFactory:
    def __init__(self):
        """
        The methods don't need to be overridden. All methods are hence Static in nature.
        No constructor needed.
        """
        pass

    @staticmethod
    def GetLoaderAdaptor(data_file=None):
        """

        :param data_file:
        :return:
        """

        assert data_file is not None, "No data_file parameter supplied"
        assert type(data_file) is str, f"Only string dtype allowed. {type(data_file)} passed instead"
        assert os.path.exists(data_file), f"Given file location {data_file} does not exist"

        loader_dict = {
            'csv': CSVLoader,
            'xlsx': '',
            'tsv': '',
            'xml': '',
            "parquet": ''
        }
        data_file_format = data_file.split('.')[1]
        assert (data_file_format in loader_dict.keys()) and (loader_dict[data_file_format] != ''),\
            f"Given file format {data_file_format} is not supported"

        loader_method = loader_dict[data_file_format]

        return loader_method