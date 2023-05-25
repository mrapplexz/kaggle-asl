import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import torch
from lion_pytorch import Lion
from madgrad import MADGRAD
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset
from torchmetrics import Metric, Accuracy, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import get_linear_schedule_with_warmup
from xztrainer import XZTrainable, BaseContext, DataType, ModelOutputsType, XZTrainer, XZTrainerConfig, SchedulerType
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from xztrainer.setup_helper import enable_tf32, set_seeds

from kekpipe.data_utils import load_points_from_pickle
from kekpipe.augms import do_augment, FILES_TO_CUT_FROM
from kekpipe.cfg import TRIM_TO_SEQ
from kekpipe.ext_att_mask import get_extended_attention_mask
from kekpipe.model import FrameEmbedder, ModelFather, XYZ_ONE_FRAME, ModelFatherWithLoss, ModelSingleTrain

import torch.nn.functional as F




class TheDataset(Dataset):
    def __init__(self, df, tgt_map, do_augment: bool, center: bool = True):
        self._df = df
        self._tgt_map = tgt_map
        self._do_augment = do_augment
        self._center = center

    def __getitem__(self, itm: int):
        itm = self._df.iloc[itm]
        outs = {
            'xyz': load_points_from_pickle(itm.path, trim=True, center=self._center)
        }
        if self._do_augment:
            outs['xyz'] = do_augment(outs['xyz'], itm['participant_id'])
        outs['frame_positions'] = torch.cat((torch.LongTensor([1]), torch.LongTensor(range(2, outs['xyz'].numel() // 3 // XYZ_ONE_FRAME + 2))))
        outs['target'] = torch.LongTensor([self._tgt_map[itm['sign']]])
        return outs

    def __len__(self):
        return len(self._df)


def collate_fn(datas):
    max_xyz = max(x['xyz'].shape[0] for x in datas)
    max_frame = max(x['frame_positions'].shape[0] for x in datas)
    xyz = torch.stack([F.pad(x['xyz'], (0, 0, 0, 0, 0, max_xyz - x['xyz'].shape[0])) for x in datas])
    frame_positions = torch.stack([F.pad(x['frame_positions'], (0, max_frame - x['frame_positions'].shape[0])) for x in datas])
    return {
        'xyz': xyz.reshape(frame_positions.shape[0], -1),
        'frame_positions': frame_positions,
        'frame_att_mask': get_extended_attention_mask(frame_positions != 0, dtype=xyz.dtype).squeeze(1).squeeze(1),
        'target': torch.cat([x['target'] for x in datas])
    }


class TheTrainable(XZTrainable):
    # def __init__(self):
    #     self._loss = CrossEntropyLoss()

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        target = data['target']
        loss, probas = context.model(
            data['xyz'],
            data['frame_positions'],
            data['frame_att_mask'],
            target
        )
        # loss = self._loss(model_out, target)
        return loss, {
            'loss': loss,
            'target': target,
            'model_out_proba': probas
        }

    def create_metrics(self) -> Dict[str, Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy': Accuracy(task='multiclass', num_classes=len(sign_map), top_k=1),
            'accuracy_top10': Accuracy(task='multiclass', num_classes=len(sign_map), top_k=10)
        }

    def update_metrics(self, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        for k in 'accuracy', 'accuracy_top10':
            metrics[k].update(model_outputs['model_out_proba'], model_outputs['target'])




if __name__ == '__main__':
    enable_tf32()
    seeds = [0xBABACAFA, 0xBEEFBEEF, 0xAD2281, 0xCAFEBA, 0xAEEBAF]
    with Path('data/sign_to_prediction_index_map.json').open('r') as f:
        sign_map = json.load(f)
    for i in range(0, 5):
        set_seeds(seeds[i])
        train = pd.read_csv('data/train.csv')
        train, val = train_test_split(train, random_state=seeds[i], test_size=0.01, stratify=train['sign'])
        FILES_TO_CUT_FROM.clear()
        for itm in train.itertuples():
            FILES_TO_CUT_FROM[itm.participant_id].append(itm.path)
        train, val = TheDataset(train, sign_map, do_augment=True), TheDataset(val, sign_map, do_augment=False)

        trainer = XZTrainer(XZTrainerConfig(
            batch_size=8,
            batch_size_eval=8,
            epochs=50,
            optimizer=lambda m: Lion(m.parameters(), lr=5e-5),
            experiment_name=f"the_ensemble_part_smaller_{i}",
            gradient_clipping=1.0,
            scheduler=lambda optim, total_steps: get_linear_schedule_with_warmup(optim, int(total_steps * 0.1),
                                                                                 total_steps),
            scheduler_type=SchedulerType.STEP,
            dataloader_num_workers=32,
            dataloader_persistent_workers=False,
            accumulation_batches=16,
            print_steps=50,
            eval_steps=500,
            save_steps=500,
            save_keep_n=10,
            collate_fn=collate_fn,
            logger=TensorboardLoggingEngineConfig()
        ), ModelSingleTrain(len(sign_map)), TheTrainable())
        trainer.train(train, val)
