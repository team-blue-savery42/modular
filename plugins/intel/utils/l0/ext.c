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
//
// lists the Level Zero driver's extensions (name and version), to see
// which record/replay or graph extensions the installed compute-runtime has.
// Build: cc -O1 -o ext ext.c -lze_loader
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS)
    return 1;
  uint32_t n = 0;
  zeDriverGet(&n, NULL);
  ze_driver_handle_t *d = calloc(n, sizeof *d);
  zeDriverGet(&n, d);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t m = 0;
    zeDriverGetExtensionProperties(d[i], &m, NULL);
    ze_driver_extension_properties_t *e = calloc(m, sizeof *e);
    zeDriverGetExtensionProperties(d[i], &m, e);
    for (uint32_t j = 0; j < m; ++j)
      printf("%s %u.%u\n", e[j].name, ZE_MAJOR_VERSION(e[j].version),
             ZE_MINOR_VERSION(e[j].version));
  }
  return 0;
}
