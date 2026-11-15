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

"""MAX's `pipelines` command line, run through utils/run-max-py.

Usage : RUN_MAX_PY_OPT=1 utils/run-max-py tests/graph/pipelines_cli.py \
    generate --model-path ~/src/intel-stack/models/Qwen3.5-0.8B --devices gpu \
    --prompt "..." --max-new-tokens 64 --top-k 1
Runs `pipelines <args>` exactly as Bazel's launcher would, with our plugins.
"""

import sys

from max._entrypoints import pipelines as cli

if __name__ == "__main__":
    sys.argv = ["pipelines", *sys.argv[1:]]
    sys.exit(cli.main())
