import pandas as pd
import numpy as np
from pathlib import Path
from carmodel_calibration.data_integration.data_set import DataSet
DATA_DIR = "/home/bookboi/Nextcloud/1_Docs/4_Master/20_Studienarbeit_paper/data/datensatz_dominik/2 (copy)/"
data_path = Path(DATA_DIR)
dtypes = {"ID": str}
for file in data_path.glob("*_WorldPositions.csv"):
    data = pd.read_csv(file, index_col=False, dtype=dtypes)
    meta_items = []
    for track_id, chunk in data.groupby(by=["ID"]):
        track_id = str(float(track_id))
        row = chunk.iloc[0]
        recording_id = int(float((file.name).split("_")[0]))
        if track_id == "654.0" and recording_id == 2:
            print("debug")
        meta_items.append({
            "trackId": track_id,
            "length": row["Length"],
            "width": row["Width"],
            "class": row["Class"],
            "recordingId": recording_id})
    meta_data = pd.DataFrame(meta_items)
    meta_file = file.parent / file.name.replace(".csv","_meta.csv")
    meta_data.to_csv(meta_file, index=False)
