from models.eff_acat import Transformer
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import itertools
import random
import pandas as pd
import math
import optuna
import torch.nn.functional as F
from optuna.samplers import TPESampler
from optuna.trial import TrialState

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


class ModelData:

    def __init__(self, enc, dec, y_true, y_id, device):
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.y_true = y_true.to(device)
        self.y_id = y_id


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    return list(random.sample(set(prod), len(prod)))


class Train:
    def __init__(self, data, args, pred_len):

        config = ExperimentConfig(pred_len, args.exp_name)
        self.data = data
        self.len_data = len(data)
        self.formatter = config.make_data_formatter()
        self.params = self.formatter.get_experiment_params()
        self.total_time_steps = self.params['total_time_steps']
        self.num_encoder_steps = self.params['num_encoder_steps']
        self.column_definition = self.params["column_definition"]
        self.pred_len = pred_len
        self.seed = args.seed
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_params = self.formatter.get_default_model_params()
        self.batch_size = self.model_params['minibatch_size'][0]
        self.attn_type = args.attn_type
        self.criterion = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.num_epochs = self.params['num_epochs']
        self.name = args.name
        self.param_history = []
        self.n_distinct_trial = 0
        self.erros = dict()
        self.exp_name = args.exp_name
        self.best_model = nn.Module()
        self.train, self.valid, self.test = self.split_data()
        self.run_optuna(args)
        self.evaluate()

    def define_model(self, d_model, n_heads,
                     stack_size, kernel, src_input_size,
                     tgt_input_size):

        stack_size, n_heads, d_model, kernel = stack_size, n_heads, d_model, kernel
        d_k = int(d_model / n_heads)

        model = Transformer(src_input_size=src_input_size,
                            tgt_input_size=tgt_input_size,
                            pred_len=self.pred_len,
                            d_model=d_model,
                            d_ff=d_model * 4,
                            d_k=d_k, d_v=d_k, n_heads=n_heads,
                            n_layers=stack_size, src_pad_index=0,
                            tgt_pad_index=0, device=self.device,
                            attn_type=self.attn_type,
                            seed=self.seed, kernel=kernel)
        model.to(self.device)

        return model

    def sample_data(self, max_samples, data):

        sample_data = batch_sampled_data(data, max_samples, self.params['total_time_steps'],
                               self.params['num_encoder_steps'], self.pred_len, self.params["column_definition"],
                                         self.seed)
        sample_data = ModelData(torch.from_numpy(sample_data['enc_inputs']),
                                                 torch.from_numpy(sample_data['dec_inputs']),
                                                 torch.from_numpy(sample_data['outputs']),
                                                 sample_data['identifier'], self.device)
        return sample_data

    def split_data(self):

        data = self.formatter.transform_data(self.data)

        train_max, valid_max = self.formatter.get_num_samples_for_calibration()
        total_num = train_max + 2 * valid_max
        train_b = int(total_num * 0.8)
        valid_len = int((total_num - train_b) / 2)

        data_sample = self.sample_data(total_num, data)

        trn_batching = batching(self.batch_size, data_sample.enc[:train_b, :, :], data_sample.dec[:train_b, :, :],
                                data_sample.y_true[:train_b, :, :], data_sample.y_id[:train_b, :, :])

        valid_batching = batching(self.batch_size, data_sample.enc[train_b:train_b + valid_len, :, :],
                                  data_sample.dec[train_b:train_b + valid_len, :, :],
                                  data_sample.y_true[train_b:train_b + valid_len, :, :],
                                  data_sample.y_id[train_b:train_b + valid_len, :, :])

        test_batching = batching(self.batch_size, data_sample.enc[-valid_len:, :, :],
                                 data_sample.dec[-valid_len:, :, :],
                                 data_sample.y_true[-valid_len:, :, :],
                                 data_sample.y_id[-valid_len:, :, :])

        trn = ModelData(trn_batching[0], trn_batching[1], trn_batching[2], trn_batching[3], self.device)
        valid = ModelData(valid_batching[0], valid_batching[1], valid_batching[2], valid_batching[3], self.device)
        test = ModelData(test_batching[0], test_batching[1], test_batching[2], test_batching[3], self.device)

        return trn, valid, test

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.name,
                                    direction="minimize", pruner=optuna.pruners.HyperbandPruner(),
                                    sampler=TPESampler(seed=1234))
        study.optimize(self.objective, n_trials=args.n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial):

        val_loss = 1e10
        src_input_size = self.train.enc.shape[3]
        tgt_input_size = self.train.dec.shape[3]
        n_batches_train = self.train.enc.shape[0]
        n_batches_valid = self.valid.enc.shape[0]

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        d_model = trial.suggest_categorical("d_model", [16, 32])
        n_heads = self.model_params['num_heads']
        stack_size = self.model_params['stack_size'][0]
        kernel = [1, 3, 6, 9] if self.attn_type == "attn_conv" else [1]
        kernel = trial.suggest_categorical("kernel", kernel)

        if [d_model, kernel] in self.param_history or self.n_distinct_trial > 4:
            raise optuna.exceptions.TrialPruned()
        else:
            self.n_distinct_trial += 1
        self.param_history.append([d_model, kernel])

        d_k = int(d_model / n_heads)

        model = Transformer(src_input_size=src_input_size,
                            tgt_input_size=tgt_input_size,
                            pred_len=self.pred_len,
                            d_model=d_model,
                            d_ff=d_model * 4,
                            d_k=d_k, d_v=d_k, n_heads=n_heads,
                            n_layers=stack_size, src_pad_index=0,
                            tgt_pad_index=0, device=self.device,
                            attn_type=self.attn_type,
                            seed=self.seed, kernel=kernel)
        model.to(self.device)

        optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 500)

        epoch_start = 0

        val_inner_loss = 1e10

        for epoch in range(epoch_start, self.num_epochs, 1):

            total_loss = 0
            for batch_id in range(n_batches_train):

                output = model(self.train.enc[batch_id], self.train.dec[batch_id])
                loss = self.criterion(output, self.train.y_true[batch_id]) + self.mae_loss(output, self.train.y_true[batch_id])

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step_and_update_lr()

            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

            model.eval()
            test_loss = 0
            for j in range(n_batches_valid):

                outputs = model(self.valid.enc[j], self.valid.dec[j])
                loss = self.criterion(outputs, self.valid.y_true[j]) + self.mae_loss(outputs, self.valid.y_true[j])
                test_loss += loss.item()

            print("val loss: {:.4f}".format(test_loss))

            if test_loss < val_inner_loss:
                val_inner_loss = test_loss
                if val_inner_loss < val_loss:
                    val_loss = val_inner_loss
                    self.best_model = model
                    torch.save({'model_state_dict': model.state_dict()},
                               os.path.join(self.model_path, "{}_{}".format(self.name, self.seed)))

        return val_loss

    def evaluate(self):

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        self.best_model.eval()
        predictions = torch.zeros(self.test.y_true.shape[0], self.test.y_true.shape[1], self.test.y_true.shape[2])
        targets_all = torch.zeros(self.test.y_true.shape[0], self.test.y_true.shape[1], self.test.y_true.shape[2])
        n_batches_test = self.test.enc.shape[0]

        for j in range(n_batches_test):

            output = self.best_model(self.test.enc[j], self.test.dec[j])
            output_map = inverse_output(output, self.test.y_true[j], self.test.y_id[j])
            p = self.formatter.format_predictions(output_map["predictions"])
            if p is not None:
                if self.exp_name == "covid":
                    tp = 'int'
                else:
                    tp = 'float'
                forecast = torch.from_numpy(extract_numerical_data(p).to_numpy().astype(tp)).to(self.device)

                predictions[j, :forecast.shape[0], :] = forecast

                targets = torch.from_numpy(extract_numerical_data(
                    self.formatter.format_predictions(output_map["targets"])).to_numpy().astype(tp)).to(self.device)

                targets_all[j, :targets.shape[0], :] = targets

        test_loss = self.criterion(predictions.to(self.device), targets_all.to(self.device)).item()
        normaliser = targets_all.to(self.device).abs().mean()
        test_loss = math.sqrt(test_loss) / normaliser

        mae_loss = self.mae_loss(predictions.to(self.device), targets_all.to(self.device)).item()
        normaliser = targets_all.to(self.device).abs().mean()
        mae_loss = mae_loss / normaliser

        print("test loss {:.4f}".format(test_loss))

        self.erros["{}_{}".format(self.name, self.seed)] = list()
        self.erros["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(test_loss)))
        self.erros["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(mae_loss)))

        error_path = "errors_{}_{}_f.json".format(self.exp_name, self.pred_len)

        if os.path.exists(error_path):
            with open(error_path) as json_file:
                json_dat = json.load(json_file)
                if json_dat.get("{}_{}".format(self.name, self.seed)) is None:
                    json_dat["{}_{}".format(self.name, self.seed)] = list()
                json_dat["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(test_loss)))
                json_dat["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(mae_loss)))

            with open(error_path, "w") as json_file:
                json.dump(json_dat, json_file)
        else:
            with open(error_path, "w") as json_file:
                json.dump(self.erros, json_file)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--attn_type", type=str, default='KittyCatConv')
    parser.add_argument("--name", type=str, default="KittyCatConv")
    parser.add_argument("--exp_name", type=str, default='covid')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--DataParallel", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_csv_path = "{}.csv".format(args.exp_name)
    raw_data = pd.read_csv(data_csv_path)

    for pred_len in [24, 48, 72]:
        Train(raw_data, args, pred_len)


if __name__ == '__main__':
    main()