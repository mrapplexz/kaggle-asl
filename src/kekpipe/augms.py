import random
from collections import defaultdict

import torch

from kekpipe.model import LEFT_HAND_IDX, RIGHT_HAND_IDX, POSE_IDX, LEFT_HAND_IDX_MIRROR, RIGHT_HAND_IDX_MIRROR, \
    POSE_IDX_MIRROR
from kekpipe.data_utils import load_points_from_pickle


def do_gaussian_noise(points: torch.Tensor):
    coef = random.uniform(0.001, 0.002)
    noise = torch.normal(0, coef, size=points.shape)
    return points + noise


def augment_hand_swap(points: torch.Tensor):
    points = points.clone()

    left_hand = points[:, LEFT_HAND_IDX].clone()
    left_hand[:, :, 0] = -left_hand[:, :, 0]  # negate x

    right_hand = points[:, RIGHT_HAND_IDX].clone()
    right_hand[:, :, 0] = -right_hand[:, :, 0]  # negate x

    pose = points[:, POSE_IDX].clone()
    pose[:, :, 0] = -pose[:, :, 0]  # negate x

    points[:, LEFT_HAND_IDX_MIRROR] = left_hand
    points[:, RIGHT_HAND_IDX_MIRROR] = right_hand
    points[:, POSE_IDX_MIRROR] = pose

    return points


def do_interpolation(points: torch.Tensor):
    new_tensor_part = [points[0]]
    for i in range(1, points.shape[0]):
        before_points = points[i - 1]
        curr_points = points[i]
        if (random.random() <= 0.15) and ((torch.isnan(before_points) != torch.isnan(curr_points)).sum() == 0):  # skip if for example hand magically DISAPPEARED or APPEARED in before or current frame...
            interpolate_at = random.uniform(0.3, 0.7)
            interpolated_points = (curr_points - before_points) * interpolate_at + before_points
            mode = random.choice(['insert'] * 6 + ['replace_before'] + ['replace_this'])
            if mode == 'insert':
                new_tensor_part.append(interpolated_points)
                new_tensor_part.append(curr_points)
            elif mode == 'replace_this':
                new_tensor_part.append(interpolated_points)
            elif mode == 'replace_before':
                new_tensor_part[-1] = interpolated_points
                new_tensor_part.append(curr_points)
        else:
            new_tensor_part.append(curr_points)
    return torch.stack(new_tensor_part, dim=0)


def do_frame_drop(points: torch.Tensor):
    keep_tensor = torch.rand(size=(points.shape[0],)) < random.uniform(0.85, 0.99)
    return points[keep_tensor]


# def augment_face_mirror(points: torch.Tensor):
#     face_idx = torch.arange(0, 467 + 1)
#     face = points[:, face_idx]
#     mirror_id = 1
#     mirror_points = points[:, mirror_id].unsqueeze(1)
#     face_new = face - mirror_points
#     face_new[:, :, 0] = -face_new[:, :, 0]  # negate x
#     face_new = face + mirror_points
#     points = points.clone()
#     points[:, face_idx] = face_new
#     return points


FILES_TO_CUT_FROM = defaultdict(list)



def do_augment(points: torch.Tensor, participant: int):
    if random.random() <= 0.5:
        mix_from = load_points_from_pickle(random.choice(FILES_TO_CUT_FROM[participant]), center=True, trim=False)
        mix_to = load_points_from_pickle(random.choice(FILES_TO_CUT_FROM[participant]), center=True, trim=False)
        max_mix_sample = min(15, int(points.shape[0] * 0.15))
        mix_frames_from = random.randint(0, max_mix_sample)
        mix_frames_to = random.randint(0, max_mix_sample)
        mix_from_start = mix_from.shape[0] - mix_frames_from
        if mix_from_start < 0:
            mix_from_start = 0
        mix_to_end = mix_frames_to
        points = torch.cat((mix_from[mix_from_start:], points, mix_to[0:mix_to_end]))
    if random.random() <= 0.85:
        points = do_gaussian_noise(points)
    if random.random() <= 0.5:
        points = augment_hand_swap(points)
    points = do_interpolation(points)
    points = do_frame_drop(points)
    # if random.random() <= 0.5:
    #     points = augment_face_mirror(points)
    return points
