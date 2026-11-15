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

// Kernels for repro.cpp. Built with:
//   clang -target spirv64 -cl-std=CL2.0 -O2 -c kernels.cl -o kernels.spv

// Writes through `address`, an integer the host passes. With an unmapped
// address the store page-faults.
kernel void store_to(ulong address, int value) {
  *(global int *)address = value;
}

// A kernel that can't fault, for the first process after a fault.
kernel void add(global const float *a, global const float *b,
                global float *out) {
  size_t i = get_global_id(0);
  out[i] = a[i] + b[i];
}
