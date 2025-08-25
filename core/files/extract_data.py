from enums import ROOT_DIR, data_path

import pandas as pd


class ExtractData:
    path: str = ROOT_DIR + data_path
    data: pd.DataFrame

    def extract_df_from_xls(self):
        self.data = pd.read_csv(self.path, index_col=0)
