"""
# Copyright (c) 2019 Wang Hanqin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

# The comparison of each optimizer is listed:
# https://github.com/mgrankin/over9000

from .adabound import AdaBound
from .adastand import Adastand
from .lookahead import Lookahead
from .novograd import Novograd
from .radam import RAdam
from .ralamb import Ralamb
from .ranger import Ranger, RangerLars