from models.eff_acat import Transformer
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import itertools
import sys
import random
import pandas as pd
import math
from data.data_loader import ExperimentConfig
from Utils.base_train import batching, batch_sampled_data, inverse_output


class NoamOpt:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


erros = dict()
config_file = dict()


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e
          , val_loss, val_inner_loss, optimizer,
          config, config_num, best_model, criterion, path):

    stop = False
    total_loss = 0
    model.train()
    for batch_id in range(train_en.shape[0]):
        output = model(train_en[batch_id], train_de[batch_id])
        loss = criterion(output, train_y[batch_id])
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

    print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

    model.eval()
    test_loss = 0
    for j in range(test_en.shape[0]):
        outputs = model(test_en[j], test_de[j])
        loss = criterion(test_y[j], outputs)
        test_loss += loss.item()

    print("val loss: {:.4f}".format(test_loss))

    if test_loss < val_inner_loss:
        val_inner_loss = test_loss
        if val_inner_loss < val_loss:
            val_loss = val_inner_loss
            best_model = model
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, "{}_{}".format(args.name, args.seed)))

        e = epoch

    if epoch - e > 10:
        stop = True

    return best_model, val_loss, val_inner_loss, stop, e


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    return list(random.sample(set(prod), len(prod)))


def evaluate(model, test_en, test_de, test_y, test_id, criterion, formatter, path, device):

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model.eval()
    mae = nn.L1Loss()
    predictions = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])

    for j in range(test_en.shape[0]):

        output = model(test_en[j], test_de[j])
        output_map = inverse_output(output, test_y[j], test_id[j])
        p = formatter.format_predictions(output_map["predictions"])
        if p is not None:
            forecast = torch.from_numpy(extract_numerical_data(p).to_numpy().astype('float32')).to(device)

            predictions[j, :forecast.shape[0], :] = forecast
            targets = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')).to(device)

            targets_all[j, :targets.shape[0], :] = targets

    test_loss = criterion(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    test_loss = math.sqrt(test_loss) / normaliser

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    mae_loss = mae_loss / normaliser

    return test_loss, mae_loss


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--attn_type", type=str, default='KittyCat')
    parser.add_argument("--name", type=str, default="KittyCat")
    parser.add_argument("--exp_name", type=str, default='electricity')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--DataParallel", type=bool, default=False)
    parser.add_argument("--pred_len", type=int, default=72)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running on GPU")

    config = ExperimentConfig(args.pred_len, args.exp_name)
    train_formatter = config.make_data_formatter()
    valid_formatter = config.make_data_formatter()
    test_formatter = config.make_data_formatter()
    formatter = test_formatter

    data_csv_path = "{}.csv".format(args.exp_name)

    print("Loading & splitting data_set...")
    raw_data = pd.read_csv(data_csv_path)

    train_b = int(len(raw_data) * 0.8)
    valid_len = int((len(raw_data) - train_b)/2)
    train_data = raw_data.iloc[:train_b, :]
    valid_data = raw_data.iloc[train_b:train_b+valid_len, :]
    test_data = raw_data.iloc[-valid_len:, :]

    train_data, valid_data, test_data = train_formatter.transform_data(train_data), \
                                        valid_formatter.transform_data(valid_data), \
                                        test_formatter.transform_data(test_data)

    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(train_data, train_max, params['total_time_steps'],
                       params['num_encoder_steps'], args.pred_len, params["column_definition"], args.seed)
    train_en, train_de, train_y, train_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    sample_data = batch_sampled_data(valid_data, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], args.pred_len, params["column_definition"], args.seed)
    valid_en, valid_de, valid_y, valid_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    sample_data = batch_sampled_data(test_data, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], args.pred_len, params["column_definition"], args.seed)
    test_en, test_de, test_y, test_id =torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    model_params = formatter.get_default_model_params()

    path = "models_{}_{}".format(args.exp_name, args.pred_len)
    if not os.path.exists(path):
        os.makedirs(path)

    criterion = nn.MSELoss()
    if args.attn_type == "conv_attn":
        kernel = [1, 3, 6, 9]
    else:
        kernel = [1]

    hyper_param = list([model_params['stack_size'],
                        [model_params['num_heads']],
                        model_params['hidden_layer_size'],
                        kernel])
    configs = create_config(hyper_param)
    print('number of config: {}'.format(len(configs)))

    val_loss = 1e10
    best_model = nn.Module()
    config_num = 0

    batch_size = model_params['minibatch_size'][0]

    train_en_p, train_de_p, train_y_p, train_id_p = batching(batch_size, train_en,
                                                             train_de, train_y, train_id)

    valid_en_p, valid_de_p, valid_y_p, valid_id_p = batching(batch_size, valid_en,
                                                             valid_de, valid_y, valid_id)

    test_en_p, test_de_p, test_y_p, test_id_p = batching(batch_size, test_en,
                                                         test_de, test_y, test_id)

    for i, conf in enumerate(configs, config_num):
        print('config {}: {}'.format(i+1, conf))

        stack_size, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)

        model = Transformer(src_input_size=train_en_p.shape[3],
                            tgt_input_size=train_de_p.shape[3],
                            d_model=d_model,
                            d_ff=d_model*4,
                            d_k=d_k, d_v=d_k, n_heads=n_heads,
                            n_layers=stack_size, src_pad_index=0,
                            tgt_pad_index=0, device=device,
                            attn_type=args.attn_type,
                            seed=args.seed, kernel=kernel)
        if args.DataParallel:
            model = nn.DataParallel(model)
        model.to(device)

        optim = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 4000)

        epoch_start = 0

        val_inner_loss = 1e10
        e = 0

        for epoch in range(epoch_start, params['num_epochs'], 1):

            best_model, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_en_p.to(device), train_de_p.to(device),
                      train_y_p.to(device), valid_en_p.to(device), valid_de_p.to(device),
                      valid_y_p.to(device), epoch, e, val_loss, val_inner_loss,
                      optim, conf, i, best_model, criterion, path)

            if stop:
                break
        print("val loss: {:.4f}".format(val_inner_loss))
        del model

    test_loss, mae_loss = evaluate(best_model, test_en_p.to(device),
                                   test_de_p.to(device), test_y_p.to(device),
                                   test_id_p, criterion, formatter, path, device)

    print("test loss {:.4f}".format(test_loss))

    erros["{}_{}".format(args.name, args.seed)] = list()
    erros["{}_{}".format(args.name, args.seed)].append(float("{:.5f}".format(test_loss)))
    erros["{}_{}".format(args.name, args.seed)].append(float("{:.5f}".format(mae_loss)))

    error_path = "errors_{}_{}.json".format(args.exp_name, args.pred_len)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get("{}_{}".format(args.name, args.seed)) is None:
                json_dat["{}_{}".format(args.name, args.seed)] = list()
            json_dat["{}_{}".format(args.name, args.seed)].append(float("{:.5f}".format(test_loss)))
            json_dat["{}_{}".format(args.name, args.seed)].append(float("{:.5f}".format(mae_loss)))

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()