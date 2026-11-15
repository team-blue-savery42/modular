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
// Round trip through device memory on one in-order immediate list:
// pinned host A -> device -> pinned host B, host synchronize, compare.
// Repeats with a new pattern each time. Argument 2 sets the copy-offload
// hint (0: copies run on the compute engine); argument 4 = 1 waits as the
// driver plugin's queue_synchronize does: a barrier signalling a
// counter-based event, then zeEventHostSynchronize on it.
// Build: cc -O1 -o copyrt copyrt.c -lze_loader
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    ze_result_t r_ = (x);                                                      \
    if (r_) {                                                                  \
      printf("%s: 0x%x\n", #x, r_);                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  size_t bytes = argc > 1 ? atol(argv[1]) : 65536;
  int offload = argc > 2 ? atoi(argv[2]) : 0;
  int reps = argc > 3 ? atoi(argv[3]) : 2000;
  int barrier = argc > 4 ? atoi(argv[4]) : 0;
  ze_init_driver_type_desc_t idesc = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                      NULL, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n = 1;
  ze_driver_handle_t drv;
  CHECK(zeInitDrivers(&n, &drv, &idesc));
  uint32_t nd = 1;
  ze_device_handle_t dev;
  CHECK(zeDeviceGet(drv, &nd, &dev));
  ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t ctx;
  CHECK(zeContextCreate(drv, &cd, &ctx));
  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = 0;
  qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER |
             (offload ? ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT : 0);
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_list_handle_t list;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &list));
  ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *a, *b, *d;
  CHECK(zeMemAllocHost(ctx, &hd, bytes, 64, &a));
  CHECK(zeMemAllocHost(ctx, &hd, bytes, 64, &b));
  CHECK(zeMemAllocDevice(ctx, &dd, bytes, 64, dev, &d));
  CHECK(zeContextMakeMemoryResident(ctx, dev, a, bytes));
  CHECK(zeContextMakeMemoryResident(ctx, dev, b, bytes));
  CHECK(zeContextMakeMemoryResident(ctx, dev, d, bytes));
  ze_event_counter_based_desc_t ed = {
      ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC};
  ed.flags = ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE |
             ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE;
  ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t ev;
  CHECK(zeEventCounterBasedCreate(ctx, dev, &ed, &ev));
#define SYNC()                                                                 \
  do {                                                                         \
    if (barrier) {                                                             \
      CHECK(zeCommandListAppendBarrier(list, ev, 0, NULL));                    \
      CHECK(zeEventHostSynchronize(ev, UINT64_MAX));                           \
    } else                                                                     \
      CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));                   \
  } while (0)
  unsigned *ua = a, *ub = b;
  size_t words = bytes / 4;
  int bad = 0, badwords = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (size_t i = 0; i < words; ++i)
      ua[i] = rep * 2654435761u + i;
    // As the HAL map: device -> host, sync, host writes, host -> device, sync.
    CHECK(zeCommandListAppendMemoryCopy(list, d, a, bytes, NULL, 0, NULL));
    SYNC();
    CHECK(zeCommandListAppendMemoryCopy(list, b, d, bytes, NULL, 0, NULL));
    SYNC();
    int w = 0;
    for (size_t i = 0; i < words; ++i)
      w += ub[i] != ua[i];
    bad += w != 0;
    badwords += w;
  }
  printf("barrier %d bytes %zu offload %d: %d of %d round trips wrong (%d "
         "words)\n",
         barrier, bytes, offload, bad, reps, badwords);
  return 0;
}
