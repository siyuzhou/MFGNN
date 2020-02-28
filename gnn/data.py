import os
import glob
import numpy as np
import torch
from . import utils


def load_data(data_path, transpose=None, edge=True, prefix='train', size=None, padding=None):
    loc_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_position*.npy')))
    vel_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_velocity*.npy')))

    all_data = []
    for loc_f, vel_f in zip(loc_files, vel_files):
        loc = np.load(loc_f)
        vel = np.load(vel_f)

        if transpose:
            loc = np.transpose(loc, transpose)
            vel = np.transpose(vel, transpose)

        data = np.concatenate([loc, vel], axis=-1).astype(np.float32)

        if size:
            data = data[:size]

        nagents = data.shape[2]
        # Add padding to nagents dim if `padding` is not None.
        if padding is not None and padding > nagents:
            pad_len = padding - nagents
            data = np.pad(data, [(0, 0), (0, 0), (0, pad_len), (0, 0)],
                          mode='constant', constant_values=0)

        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)

    if edge:
        edge_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_edge*.npy')))

        all_edges = []
        for edge_f in edge_files:
            # Edge data.
            edge_data = np.load(edge_f).astype(np.int)

            if size:
                edge_data = edge_data[:size]

            # Padding
            nagents = edge_data.shape[1]
            if padding is not None and padding > nagents:
                pad_len = padding - nagents
                edge_data = np.pad(edge_data, [(0, 0), (0, pad_len), (0, pad_len)],
                                   mode='constant', constant_values=0)

            all_edges.append(edge_data)

        all_edges = np.concatenate(all_edges, axis=0)

        return all_data, all_edges

    return (all_data,)


def preprocess_data(data, seg_len, pred_steps, edge_type=1):
    time_series, edges = data
    time_steps, nagents, ndims = time_series.shape[1:]

    edge_label = edge_type + 1 # Accounting for "no connection"

    # time_series shape [num_sims, time_steps, nagents, ndims]
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, nagents, ndims]
    time_segs_stack = utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                              seg_len)
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, nagents, ndims]
    expected_time_segs_stack = utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                       pred_steps)
    assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
            == time_steps - seg_len - pred_steps + 1)

    time_segs = torch.from_numpy(time_segs_stack.reshape([-1, seg_len, nagents, ndims]))
    expected_time_segs = torch.from_numpy(expected_time_segs_stack.reshape([-1, pred_steps, nagents, ndims]))

    edges_one_hot = utils.one_hot(edges, edge_label, np.float32)
    edges_one_hot = torch.from_numpy(np.repeat(edges_one_hot, time_segs_stack.shape[1], axis=0))

    return [time_segs, edges_one_hot], expected_time_segs


