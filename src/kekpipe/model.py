from typing import Optional, List

import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn, Tensor, LongTensor
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler
import torch.nn.functional as F

from kekpipe.cfg import XYZ_ONE_FRAME, MAX_POSITIONS

EMBED_DIM_EVENT = 384


class PoolerFinal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, att_mask: Optional[Tensor]) -> torch.Tensor:
        if att_mask is not None:
            att_mask_flag = att_mask.squeeze(1).squeeze(1) == 0
            pooled = (att_mask_flag.unsqueeze(2) * hidden_states).sum(dim=1) / att_mask_flag.sum(dim=1).unsqueeze(1)
        else:
            pooled = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(pooled)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def calc_n_encoder_points(n_points):
    return n_points * 2 + n_points * 2 + n_points * 2 + 4 * 2


def _get_delta(bs, x, n_points: Optional[int] = None):
    if bs > 1:
        x_next = x[:, 1:]
        x_curr = x[:, :-1]
    else:
        x_next = x.reshape(-1)
        x_curr = x.reshape(-1)
        if n_points is not None:
            x_next = x_next[2 * n_points:x_next.numel()].reshape(1, -1, n_points, 2)
            x_curr = x_curr[:x_curr.numel() - (2 * n_points)].reshape(1, -1, n_points, 2)
        else:
            x_next = x_next[2:x_next.numel()].reshape(1, -1, 2)
            x_curr = x_curr[:x_curr.numel() - 2].reshape(1, -1, 2)

    if n_points is None:
        zeros_shape = (bs, 1, 2)
    else:
        zeros_shape = (bs, 1, n_points, 2)

    delta = torch.cat((
        torch.zeros(zeros_shape, device=x_curr.device),
        x_next - x_curr
    ), dim=1)
    return delta


class PointSetFeatureExtractor(nn.Module):
    def __init__(self, n_points):
        super().__init__()
        self._n_points = n_points

    def _nan_to_num(self, x: Tensor, to_num_nan: float, copy: bool = False):
        if copy:
            x = x.clone()
        x[torch.isnan(x)] = to_num_nan
        x[(x == torch.inf) | (x == -torch.inf)] = to_num_nan
        return x

    def forward(self, x):
        bs = x.shape[0]

        points_velocity = _get_delta(bs, x, self._n_points)
        points_acceleration = _get_delta(bs, points_velocity, self._n_points)
        points_velocity = points_velocity.reshape(bs, -1, self._n_points * 2)
        points_acceleration = points_acceleration.reshape(bs, -1, self._n_points * 2)
        self._nan_to_num(points_velocity, 0)
        self._nan_to_num(points_acceleration, 0)

        points_max = self._nan_to_num(x, -torch.inf, copy=True).max(dim=2).values
        points_min = self._nan_to_num(x, torch.inf, copy=True).min(dim=2).values
        points_scale = points_max - points_min
        self._nan_to_num(points_scale, 1)
        points_shift = (points_max - points_min) / 2 + points_min
        self._nan_to_num(points_shift, 0)

        x = (x - points_shift.reshape(bs, -1, 1, 2)) / points_scale.reshape(bs, -1, 1, 2)
        self._nan_to_num(x, 0)

        points_movement = _get_delta(bs, points_shift)
        points_movement_scale = _get_delta(bs, points_movement)

        x = x.reshape(bs, -1, self._n_points * 2)
        x = torch.cat((x, points_velocity, points_movement, points_acceleration, points_shift, points_movement_scale,
                       points_scale), dim=2)
        x = x.reshape(-1, calc_n_encoder_points(self._n_points))
        return x


class PointSetFeatureEmbedder(nn.Module):
    def __init__(self, n_points):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Linear(calc_n_encoder_points(n_points), EMBED_DIM_EVENT * 2, bias=True),
            nn.LayerNorm(EMBED_DIM_EVENT * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(EMBED_DIM_EVENT * 2, EMBED_DIM_EVENT, bias=True),
            nn.LayerNorm(EMBED_DIM_EVENT),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, bs):
        x = self._encoder(x)
        x = x.view(bs, -1, x.shape[-1])
        return x


POSE_IDX = torch.LongTensor([
    489,  # nose
    490,  # left_eye_inner
    491,  # left_eye
    492,  # left_eye_outer
    493,  # right_eye_inner
    494,  # right_eye
    495,  # right_eye_outer
    496,  # left_ear
    497,  # right_ear
    498,  # mouth_left
    499,  # mouth_right
    500,  # left_shoulder
    501,  # right_shoulder
    502,  # left_elbow
    503,  # right_elbow
    504,  # left_wrist
    505,  # right_wrist
    506,  # left_pinky
    507,  # right_pinky
    508,  # left_index
    509,  # right_index
    510,  # left_thumb
    511,  # right_thumb
    512,  # left_hip
    513,  # right_hip
    514,  # left_knee
    515,  # right_knee
    516,  # left_ankle
    517,  # right_ankle
    518,  # left_heel
    519,  # right_heel
    520,  # left_foot_index
    521  # right_foot_index
])
POSE_IDX_MIRROR = torch.LongTensor([
    489,  # nose
    493,  # right_eye_inner
    494,  # right_eye
    495,  # right_eye_outer
    490,  # left_eye_inner
    491,  # left_eye
    492,  # left_eye_outer
    497,  # right_ear
    496,  # left_ear
    499,  # mouth_right
    498,  # mouth_left
    501,  # right_shoulder
    500,  # left_shoulder
    503,  # right_elbow
    502,  # left_elbow
    505,  # right_wrist
    504,  # left_wrist
    507,  # right_pinky
    506,  # left_pinky
    509,  # right_index
    508,  # left_index
    511,  # right_thumb
    510,  # left_thumb
    513,  # right_hip
    512,  # left_hip
    515,  # right_knee
    514,  # left_knee
    517,  # right_ankle
    516,  # left_ankle
    519,  # right_heel
    518,  # left_heel
    521,  # right_foot_index
    520  # left_foot_index
])
LEFT_HAND_IDX = torch.arange(468, 488 + 1, dtype=torch.long)
RIGHT_HAND_IDX = torch.arange(522, 542 + 1, dtype=torch.long)
LEFT_HAND_IDX_MIRROR = RIGHT_HAND_IDX
RIGHT_HAND_IDX_MIRROR = LEFT_HAND_IDX
LIPS_IDX = torch.LongTensor([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308
])


class FrameEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self._lips_embed = PointSetFeatureEmbedder(LIPS_IDX.shape[0])
        self._pose_embed = PointSetFeatureEmbedder(POSE_IDX.shape[0])
        self._left_hand_embed = PointSetFeatureEmbedder(LEFT_HAND_IDX.shape[0])
        self._right_hand_embed = PointSetFeatureEmbedder(RIGHT_HAND_IDX.shape[0])
        self._embed_weights = nn.Parameter(torch.empty((4, 1, 1, 1)))
        nn.init.normal_(self._embed_weights)

    def forward(self, lips_fts, pose_fts, left_hand_fts, right_hand_fts, bs):
        lips_emb = self._lips_embed(lips_fts, bs)
        pose_emb = self._pose_embed(pose_fts, bs)
        left_hand_emb = self._left_hand_embed(left_hand_fts, bs)
        right_hand_emb = self._right_hand_embed(right_hand_fts, bs)
        x = torch.stack((left_hand_emb, lips_emb, pose_emb, right_hand_emb))
        x = (x * self._embed_weights).sum(dim=0)
        return x


class SequenceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = RobertaConfig(
            hidden_size=EMBED_DIM_EVENT,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=EMBED_DIM_EVENT * 4
        )
        self._cls_vector = nn.Parameter(torch.empty(EMBED_DIM_EVENT))
        nn.init.normal_(self._cls_vector)
        self._pos_encoder = nn.Embedding(MAX_POSITIONS, EMBED_DIM_EVENT)
        self._encoder = RobertaEncoder(cfg)
        self._encoder_norm = nn.LayerNorm(EMBED_DIM_EVENT, cfg.layer_norm_eps)
        self._encoder_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self._encoder_pooler = PoolerFinal(cfg)

        self._prepend_position = nn.Parameter(torch.LongTensor([1]), requires_grad=False)
        self._prepend_att_mask = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    def _unsqueeze(self, x, frame_vectors):
        return x.unsqueeze(0).repeat((frame_vectors.shape[0], 1))

    def forward(self, frame_vectors, frame_positions, frame_att_mask: Optional[Tensor]):
        if frame_att_mask is not None:
            frame_att_mask = frame_att_mask.unsqueeze(1).unsqueeze(1)
        frame_prepend = self._unsqueeze(self._cls_vector, frame_vectors).unsqueeze(1)
        frame_vectors = torch.cat((frame_prepend, frame_vectors), dim=1)
        frame_vectors = self._encoder_norm(frame_vectors)
        frame_vectors = self._encoder_dropout(frame_vectors)
        frame_vectors = frame_vectors + self._pos_encoder(frame_positions)
        frame_vectors = self._encoder(frame_vectors, attention_mask=frame_att_mask).last_hidden_state
        frame_vectors = self._encoder_pooler(frame_vectors, frame_att_mask)
        return frame_vectors


class ModelFather(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self._frame = FrameEmbedder()
        self._seq = SequenceEmbedder()
        self._cls_drop = nn.Dropout(0.1)
        self._cls_head = nn.Linear(EMBED_DIM_EVENT, n_out)

    def forward(self, features: List[Tensor], frame_positions: Tensor, frame_att_mask: Optional[Tensor]):
        t = self._frame(*features, bs=frame_positions.shape[0])
        t = self._seq(t, frame_positions, frame_att_mask)
        return t


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._lips_idx = nn.Parameter(LIPS_IDX, requires_grad=False)
        self._lips_ext = PointSetFeatureExtractor(LIPS_IDX.shape[0])
        self._pose_idx = nn.Parameter(POSE_IDX, requires_grad=False)
        self._pose_ext = PointSetFeatureExtractor(POSE_IDX.shape[0])
        self._left_hand_idx = nn.Parameter(LEFT_HAND_IDX, requires_grad=False)
        self._left_hand_ext = PointSetFeatureExtractor(LEFT_HAND_IDX.shape[0])
        self._right_hand_idx = nn.Parameter(RIGHT_HAND_IDX, requires_grad=False)
        self._right_hand_ext = PointSetFeatureExtractor(RIGHT_HAND_IDX.shape[0])

    def forward(self, xyz: Tensor, frame_positions: Tensor):
        xyz[torch.isnan(xyz)] = 0
        xyz = xyz.reshape(frame_positions.shape[0], -1, XYZ_ONE_FRAME, 3)
        xyz = xyz[:, :, :, 0:2]
        lips_fts = self._lips_ext(xyz.index_select(dim=2, index=self._lips_idx))
        pose_fts = self._pose_ext(xyz.index_select(dim=2, index=self._pose_idx))
        left_hand_fts = self._left_hand_ext(xyz.index_select(dim=2, index=self._left_hand_idx))
        right_hand_fts = self._right_hand_ext(xyz.index_select(dim=2, index=self._right_hand_idx))
        return lips_fts, pose_fts, left_hand_fts, right_hand_fts


class FusedArcFace(nn.Module):
    def __init__(self, module: ArcFaceLoss):
        super().__init__()
        self.w = nn.Parameter(F.normalize(module.W.data.t(), dim=1, p=2).t())

    def get_logits(self, embeddings):
        query_emb_normalized = F.normalize(embeddings, dim=1, p=2)
        logits = torch.matmul(query_emb_normalized, self.w)
        return logits * 64


class ModelFatherWithLoss(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self._father = ModelFather(n_out)
        self._arcface = ArcFaceLoss(num_classes=n_out, embedding_size=EMBED_DIM_EVENT)

    def forward(self, features: List[Tensor], frame_positions: Tensor, frame_att_mask: Optional[Tensor], target: Optional[Tensor]):
        x = self._father(features, frame_positions, frame_att_mask)
        if target is not None:
            loss = self._arcface(x, target)
            return loss, torch.softmax(self._arcface.get_logits(x), dim=-1)
        else:
            return torch.softmax(self._arcface.get_logits(x), dim=-1)


class ModelSingleTrain(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.father = ModelFatherWithLoss(n_out)

    def forward(self, xyz: torch.Tensor, frame_positions: Tensor, frame_att_mask: Optional[Tensor], target: Optional[Tensor]):
        fts = self.encoder(xyz, frame_positions)
        return self.father(fts, frame_positions, frame_att_mask, target)


class ModelEnsemble(nn.Module):
    def __init__(self, n_out, n_predictors):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.fathers = nn.ModuleList([ModelFatherWithLoss(n_out) for _ in range(n_predictors)])

    def fuse(self):
        for father in self.fathers:
            father._arcface = FusedArcFace(father._arcface)

    def forward(self, xyz: torch.Tensor, frame_positions: Tensor, frame_att_mask: Optional[Tensor], target: Optional[Tensor]):
        fts = self.encoder(xyz, frame_positions)
        outputs = []
        for father in self.fathers:
            x = father(fts, frame_positions, frame_att_mask, target)
            outputs.append(x)
        return torch.mean(torch.stack(outputs, dim=0), dim=0)
