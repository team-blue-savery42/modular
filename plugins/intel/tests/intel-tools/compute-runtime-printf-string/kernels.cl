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

// Kernels that print a string with printf's %s. Built with:
//   clang -target spirv64 -cl-std=CL2.0 -O2 -c kernels.cl -o kernels.spv
// Each takes the one int buffer utils/l0/run.cpp passes (p[0] = 7).

// The argument is a string literal: IGC places it where compute-runtime
// finds it on the host.
kernel void literal_string(global int *p) {
  printf("literal: %s\n", "hello");
}

// The argument is a string built at run time in the kernel's buffer. The
// buffer is shared memory, so its device address is also a host address.
kernel void shared_buffer_string(global int *p) {
  global char *s = (global char *)(p + 1);
  s[0] = 'h';
  s[1] = 'i';
  s[2] = 0;
  printf("shared buffer: %s\n", s);
}

// The argument is a string built at run time in private memory, whose
// address means nothing on the host.
kernel void private_string(global int *p) {
  char s[4];
  s[0] = (char)('a' + p[0]);
  s[1] = 'i';
  s[2] = 0;
  printf("private: %s\n", s);
  p[1] = s[0];
}
