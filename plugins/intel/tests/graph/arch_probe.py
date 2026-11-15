# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Sean Avery. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Prints which MAX architecture each task resolves for an HF architecture
name (Qwen3-8B resolves to the embedding pipeline without
--no-prefer-module-v3).
Usage: utils/run-max-py tests/graph/arch_probe.py NAME"""

import sys

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.modeling.types import PipelineTask

name = sys.argv[1]
for task in (
    None,
    PipelineTask.TEXT_GENERATION,
    PipelineTask.EMBEDDINGS_GENERATION,
):
    a = PIPELINE_REGISTRY.retrieve_architecture(name, task=task)
    print(
        task,
        a and a.name,
        a and a.task,
        a and getattr(a, "pipeline_model", None),
    )
lk = PIPELINE_REGISTRY._arch_lookup
print(
    {
        k: (v.task, getattr(v, "pipeline_model", None))
        for k, v in lk._architectures_by_task.items()
        if k[0] == name
    }
)
