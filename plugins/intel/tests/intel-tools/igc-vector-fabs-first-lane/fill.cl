//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Sean Avery. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

// Leaves +inf in many registers: loads four float16 per work-item (all +inf
// in the input), keeps them live together, stores a sum of them.
kernel void fill(global const float16 *in, global float16 *out) {
  size_t i = get_global_id(0);
  float16 a = in[4 * i], b = in[4 * i + 1], c = in[4 * i + 2], d = in[4 * i + 3];
  out[i] = (a + b) * (c + d) + a * d - b * c;
}
