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

"""Samples the B70's used device memory through Level Zero Sysman.

Calls zesInit, zesDriverGet, zesDeviceGet, zesDeviceEnumMemoryModules and
zesMemoryGetState (/usr/include/level_zero/zes_api.h) through ctypes, and
every PERIOD seconds prints `<time.monotonic_ns()> <used bytes>` (size - free
summed over the device's memory modules: device-wide, every process) until
killed or until STOPFILE exists. The clock is the one bench_model.py --marks
and the driver plugin's profile use, so samples line up with steps.

Usage: python3 sysman_mem.py [PERIOD=0.5] [STOPFILE]
"""

import ctypes
import os
import sys
import time

ZES_STRUCTURE_TYPE_MEM_STATE = 0x1E


class MemState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_int),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


def check(result, what):
    if result != 0:
        raise SystemExit(f"{what} failed: 0x{result:x}")


def modules():
    ze = ctypes.CDLL("libze_loader.so.1")
    check(ze.zesInit(0), "zesInit")
    count = ctypes.c_uint32(0)
    check(ze.zesDriverGet(ctypes.byref(count), None), "zesDriverGet")
    drivers = (ctypes.c_void_p * count.value)()
    check(ze.zesDriverGet(ctypes.byref(count), drivers), "zesDriverGet")
    found = []
    for driver in drivers:
        n = ctypes.c_uint32(0)
        check(
            ze.zesDeviceGet(ctypes.c_void_p(driver), ctypes.byref(n), None),
            "zesDeviceGet",
        )
        devices = (ctypes.c_void_p * n.value)()
        check(
            ze.zesDeviceGet(ctypes.c_void_p(driver), ctypes.byref(n), devices),
            "zesDeviceGet",
        )
        for device in devices:
            m = ctypes.c_uint32(0)
            check(
                ze.zesDeviceEnumMemoryModules(
                    ctypes.c_void_p(device), ctypes.byref(m), None
                ),
                "zesDeviceEnumMemoryModules",
            )
            mems = (ctypes.c_void_p * m.value)()
            check(
                ze.zesDeviceEnumMemoryModules(
                    ctypes.c_void_p(device), ctypes.byref(m), mems
                ),
                "zesDeviceEnumMemoryModules",
            )
            found += [ctypes.c_void_p(h) for h in mems]
    return ze, found


def main():
    period = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    stop = sys.argv[2] if len(sys.argv) > 2 else None
    ze, mems = modules()
    state = MemState()
    while stop is None or not os.path.exists(stop):
        used = 0
        for mem in mems:
            state.stype = ZES_STRUCTURE_TYPE_MEM_STATE
            state.pNext = None
            check(
                ze.zesMemoryGetState(mem, ctypes.byref(state)),
                "zesMemoryGetState",
            )
            used += state.size - state.free
        print(time.monotonic_ns(), used, flush=True)
        time.sleep(period)


if __name__ == "__main__":
    main()
