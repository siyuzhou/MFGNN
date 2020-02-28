import os
import sys
import argparse
import json

import torch
import numpy as np
import tqdm

import mpnn
from mpnn.data import load_data, preprocess_data


def eval_base_line(eval_data):
    time_series = eval_data[0]
    return np.mean(np.square(time_series[:, :-1, :, :] -
                             time_series[:, 1:, :, :]))


def save_model(model, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(log_dir, "mpnn.pth"))


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, "mpnn.pth")
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))


def load_params(config, data):
    with open(config) as f:
        model_params = json.load(f)

    time_series = data[0]
    nagents, ndims = time_series.shape[-2:]

    filters = model_params['cnn']['filters']
    seg_len = 2 * len(filters) + 1
    input_size = ndims if not filters else filters[-1]

    model_params.update({'num_nodes': nagents, 
                         'ndims': ndims,
                         'pred_steps': ARGS.pred_steps, 
                         'time_seg_len': seg_len})

    model_params['edge_encoder']['input_size'] = input_size * 2 # Because [sources, targets]
    model_params['node_decoder']['input_size'] = \
        input_size + model_params['edge_encoder']['hidden_units'][-1]

    return model_params


def train(model, data_loader, learning_rate, epochs, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss(reduction='mean')

    model.train()

    train_losses = []
    for e in range(epochs):
        data_progress = tqdm.tqdm(data_loader, desc=f'Epoch {e}')
        for i, (time_segs, edges, expected_time_segs) in enumerate(data_progress):
            time_segs, edges, expected_time_segs = \
                time_segs.to(device), edges.to(device), expected_time_segs.to(device)

            optimizer.zero_grad()
            
            prediction = model(time_segs, edges)

            loss = mse(prediction, expected_time_segs)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            data_progress.set_postfix(loss=f'{sum(train_losses[-100:])/100:0.5f}')

        save_model(model, ARGS.log_dir)
        

def eval(model, data_loader, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    mse = torch.nn.MSELoss(reduction='mean')

    eval_losses = []
    data_progress = tqdm.tqdm(data_loader, desc=f'Evaluation')
    for i, (time_segs, edges, expected_time_segs) in enumerate(data_progress):
        time_segs, edges, expected_time_segs = \
            time_segs.to(device), edges.to(device), expected_time_segs.to(device)

        with torch.no_grad():
            prediction = model(time_segs, edges)
            loss = mse(prediction, expected_time_segs)

        eval_losses.append(loss.item())

        data_progress.set_postfix(avg_loss=f'{sum(eval_losses)/len(eval_losses):0.5f}')

    return avg_loss


def test(model, data_loader, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []
    data_progress = tqdm.tqdm(data_loader, desc='Test')
    for i, (time_segs, edges, expected_time_segs) in enumerate(data_progress):
        time_segs, edges, expected_time_segs = \
            time_segs.to(device), edges.to(device), expected_time_segs.to(device)

        with torch.no_grad():
            prediction = model(time_segs, edges)

        predictions.append(prediction.cpu().numpy())

    return np.concatenate(predictions, axis=0)
    

def main():
    if ARGS.train:
        prefix = 'train'
    elif ARGS.eval:
        prefix = 'valid'
    elif ARGS.test:
        prefix = 'test'

    data = load_data(ARGS.data_dir, ARGS.data_transpose,
                     prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding)

    model_params = load_params(ARGS.config, data)

    (time_segs, edges), expected_time_segs = preprocess_data(
        data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'])
    print(f"\nData from {ARGS.data_dir} processed.\n")

    dataset = torch.utils.data.TensorDataset(time_segs, edges, expected_time_segs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=ARGS.train)

    print("Building model...")
    
    model = mpnn.MPNN(model_params)
    load_model(model, ARGS.log_dir)

    # Train, evaluate or test.
    if ARGS.train:
        train(model, loader, model_params['learning_rate'], ARGS.epochs, ARGS.device)

    elif ARGS.eval:
        result = eval(model, loader, ARGS.device)
        print("Evaluating baseline...")
        baseline = eval_base_line(data)
        print('Baseline:', baseline, '\t| MSE / Baseline:', result / baseline)

    elif ARGS.test:
        prediction = test(model, loader, ARGS.device)
        np.save(os.path.join(ARGS.log_dir, f'prediction_{ARGS.pred_steps}.npy'), prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device on which to run the model')
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--data-size', type=int, default=None,
                        help='optional data size cap to use for training')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--max-padding', type=int, default=None,
                        help='max pad length to the number of agents dimension')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='turn on evaluation')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    main()