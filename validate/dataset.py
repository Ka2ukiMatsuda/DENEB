import pandas as pd
import itertools
from deneb.models.model_base import CVPRDataset

def get_dataset(path):
    df = pd.read_csv(path)
    df = df[["mt","refs","score", "imgid"]]
    refs_list = []
    for refs in df["refs"]:
        refs = eval(refs)
        refs_list.append(refs)

    df["refs"] = refs_list
    df["mt"] = df["mt"].astype(str)
    df["score"] = df["score"].astype(float)
    df["imgid"] = df["imgid"].astype(str)
    test_dataset = df.to_dict("records")
    test_dataset = CVPRDataset(test_dataset, "data_en/images")
    return test_dataset
