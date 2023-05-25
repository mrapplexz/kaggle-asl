import pandas as pd
import torch

from kekpipe.cfg import XYZ_ONE_FRAME, TRIM_TO_SEQ


def center_nose(xyz):
    mirror_id = 1
    center_points = xyz[:, mirror_id].unsqueeze(1)  # mirror relative to nose tip
    return xyz - center_points


def to_point_tensors(x, center: bool):
    xyz = torch.Tensor([x['x'], x['y'], x['z']]).T.view(-1, XYZ_ONE_FRAME, 3)
    if center:
        xyz = center_nose(xyz)

    return xyz


def load_points_from_pickle(file, trim: bool, center: bool):
    pqt = pd.read_parquet('data/' + file, engine='fastparquet')
    if trim:
        pqt = pqt.iloc[:XYZ_ONE_FRAME * TRIM_TO_SEQ]
    return to_point_tensors(pqt, center)