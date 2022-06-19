# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from Utils import base
from data import electricity


ElectricityFormatter = electricity.ElectricityFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class AirQualityFormatter(ElectricityFormatter):

    _column_definition = [
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('NO2', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('CO', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('TEMP', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
    ]

    def split_data(self, df, valid_boundary=1260, test_boundary=1360):
        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 50,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': [16, 32],
            'learning_rate': 0.001,
            'minibatch_size': [256],
            'max_gradient_norm': 100.,
            'num_heads': 8,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data_set for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 128000, 16384