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

// A kernel that does nothing but write one word, for a Level Zero program
// to load with the validation layer on. Built with:
//   clang -target spirv64 -cl-std=CL2.0 -O2 -c kernels.cl -o kernels.spv
kernel void store_one(global int *p) { p[0] = 1; }
