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
