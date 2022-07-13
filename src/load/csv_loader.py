import pandas as pd
class CSVLoader:

    def __init__(self, data_file):
        self._data_file = data_file

    def load_data(self):
        """

        :return: load data into as pandas df
        """
        df = pd.read_csv(self._data_file)
        return df