import json
from pathlib import Path

import pandas as pd
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from torch import nn
from tqdm import tqdm

from kekpipe.cfg import TRIM_TO_SEQ, MAX_POSITIONS
from kekpipe.data_utils import center_nose
from kekpipe.model import ModelFatherWithLoss, XYZ_ONE_FRAME, ModelEnsemble
from kekpipe.train import TheDataset, collate_fn


class ModelAdapted(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        pos_ids = torch.arange(2, MAX_POSITIONS)
        self._pos_ids = nn.Parameter(pos_ids, requires_grad=False)

    def forward(self, inputs):
        xyz = center_nose(inputs)
        xyz = xyz.reshape(1, -1, XYZ_ONE_FRAME, 3)
        xyz = xyz.reshape(1, -1)
        xyz = xyz[:, 0:TRIM_TO_SEQ * XYZ_ONE_FRAME * 3]
        cut_to = torch.clip(inputs.shape[0], max=TRIM_TO_SEQ)
        pos_ids = self._pos_ids[0:cut_to]
        pos_ids = pos_ids.repeat(1).reshape(1, -1).transpose(0, 1).flatten()
        pos_ids = torch.cat((torch.LongTensor([1]), pos_ids))
        pos_ids = pos_ids.unsqueeze(0)
        outputs = self._model(xyz, pos_ids, None, None)
        return outputs.reshape(-1)


def process_weight(k, v):
    if k == '_father._seq._pos_encoder.weight':
        return v[:MAX_POSITIONS, :]
    return v



if __name__ == '__main__':
    with Path('data/sign_to_prediction_index_map.json').open('r') as f:
        sign_map = json.load(f)

    ensemble_parts = (3, 1, 2)

    weights = [{(k if not k.startswith('father.') else k[len('father.'):]): process_weight(k, v) for k, v in torch.load(f'checkpoint/the_ensemble_part_small_{i}/save-36550.pt')['model'].items() if v.dtype != torch.long} for i in ensemble_parts]  # don't load ids
    mdl = ModelEnsemble(len(sign_map), len(ensemble_parts))
    for i in range(len(ensemble_parts)):
        print(mdl.fathers[i].load_state_dict(weights[i], strict=False))
    mdl.fuse()
    mdl = ModelAdapted(mdl)
    mdl = mdl.eval()
    train = pd.read_csv('data/train.csv')
    data_itm = collate_fn([TheDataset(train, sign_map, do_augment=False, center=False)[0]])

    with torch.inference_mode():
        torch.onnx.export(mdl, (data_itm['xyz'].squeeze(0).reshape(-1, XYZ_ONE_FRAME, 3),), 'out.onnx',
                          input_names=['inputs'],
                          output_names=['outputs'],
                          dynamic_axes={
                              'inputs': {0: 'frame'},
                          }
                          )
        print(123)
    opts = SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    # opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # opts.optimized_model_filepath = 'out.onnx'
    sess = InferenceSession('out.onnx', opts)
    # for i in tqdm(range(300)):
    outs = sess.run(['outputs'], input_feed={'inputs': data_itm['xyz'].squeeze(0).reshape(-1, XYZ_ONE_FRAME, 3).numpy()})
    print(sess.end_profiling())
    print(outs)
