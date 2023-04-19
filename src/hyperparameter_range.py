"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Hyperparameter range specification.
"""

from random import uniform

hp_range = {
    "beta": [0., .01, .02, .05, .1],
    "emb_dropout_rate": [0, .1, .2, .3],
    "ff_dropout_rate": [0, .1, .2, .3],
    "action_dropout_rate": [.95],
    "bandwidth": [200, 256, 400, 512],
    "relation_only": [True, False],
    "learning_rate": [uniform(1e-4, 1e-3) for _ in range(10)],
    "weight_decay": [0.02, 0.03, 0.01]
}
