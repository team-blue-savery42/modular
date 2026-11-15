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
// does a barrier's counter-based event on an in-order immediate list
// with the copy-offload hint (the driver plugin's queue, Context.cpp) complete
// only after a host-to-device copy appended before it, which the hint sends
// to the copy engine? The plugin's free marks rely on it. Appends a large
// pinned-host-to-device copy that signals event C, then a barrier signalling
// event M (as markLocked does), and polls both: a mark that completes while
// the copy is still running means memory could be released under it.
// Build: cc -O1 -o markorder markorder.c -lze_loader
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    ze_result_t r_ = (x);                                                      \
    if (r_) {                                                                  \
      printf("%s: 0x%x\n", #x, r_);                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static ze_event_handle_t counter_event(ze_context_handle_t ctx,
                                       ze_device_handle_t dev) {
  ze_event_counter_based_desc_t d = {
      ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC};
  d.flags = ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE |
            ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE;
  d.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  d.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t e = NULL;
  if (zeEventCounterBasedCreate(ctx, dev, &d, &e))
    return NULL;
  return e;
}

int main(int argc, char **argv) {
  size_t bytes = (argc > 1 ? atol(argv[1]) : 1024) << 20;
  int offload = argc > 2 ? atoi(argv[2]) : 1;
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
  void *host, *devp;
  CHECK(zeMemAllocHost(ctx, &hd, bytes, 64, &host));
  CHECK(zeMemAllocDevice(ctx, &dd, bytes, 64, dev, &devp));
  memset(host, 1, bytes);
  ze_event_handle_t c = counter_event(ctx, dev), m = counter_event(ctx, dev);
  if (!c || !m) {
    printf("no counter-based events\n");
    return 1;
  }
  int early = 0;
  for (int rep = 0; rep < 5; ++rep) {
    double t0 = now_ms();
    CHECK(zeCommandListAppendMemoryCopy(list, devp, host, bytes, c, 0, NULL));
    CHECK(zeCommandListAppendBarrier(list, m, 0, NULL));
    double tm = -1, tc = -1;
    while (tm < 0 || tc < 0) {
      if (tm < 0 && zeEventQueryStatus(m) == ZE_RESULT_SUCCESS)
        tm = now_ms() - t0;
      if (tc < 0 && zeEventQueryStatus(c) == ZE_RESULT_SUCCESS)
        tc = now_ms() - t0;
    }
    printf("rep %d: %zu MiB copy done at %.2f ms, barrier mark at %.2f ms%s\n",
           rep, bytes >> 20, tc, tm,
           tm + 0.05 < tc ? "  <-- MARK BEFORE COPY" : "");
    early += tm + 0.05 < tc;
    CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
  }
  printf("offload hint %d: %d of 5 marks completed before the copy\n", offload,
         early);
  return 0;
}
