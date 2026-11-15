# Licensing of the Intel GPU changes

All new code in `plugins/intel/`, and every new or changed file outside
the directories below, is under the Apache License v2.0 with LLVM
Exceptions and carries Modular's standard header for it. New files
written for this work name Sean Avery as the copyright holder. The new
standard-library and MAX target plugins (`Mojo/stdlib/std/_plugin/intel/`,
`max/mojo/max/_plugin/intel/`) follow the other vendors' plugins beside
them and name both Modular and Sean Avery. Changed upstream files keep
Modular's copyright line.

The following files are under the MAX License
(<https://modular.com/legal/max-license>): files in a `max_license/`
directory, and the MLRT and GraphCompiler sources. The changes to them
are limited to the Intel (`level_zero`) path; no other device's
behaviour changes.

| File                                                                            | Change                                                                                                                                                                                                      | Kind                                                                           |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `GraphCompiler/lib/MEFModel/MEFModel.cpp`                                       | `captureStreams` also takes HAL devices whose API is `level_zero`, so device graph capture reaches Intel GPUs in the HAL build                                                                              | modified                                                                       |
| `MLRT/include/MLRT/Driver/DeviceContext/HALDeviceContext.h`                     | `DeviceGraph` holds a plugin graph, its queue and the plugin's destroy and launch entry points                                                                                                              | modified                                                                       |
| `MLRT/lib/Driver/DeviceContext/HAL/HALDeviceContext.cpp`                        | `DeviceGraph` capture and replay through a plugin's graph entry points (outside the HAL C API), resolved with `dlsym` from the stream's plugin; plugins without them get the existing "not supported" error | modified                                                                       |
| `MLRT/lib/Driver/DeviceContext/HAL/hal_device_context_exports.mojo`             | new export `MojoHAL_DeviceStream_pluginQueue`: a stream's plugin queue, driver and library path for the C++ shim                                                                                            | modified                                                                       |
| `max/mojo/max/_driver/max_license/device_context_dispatch/device_function.mojo` | `enqueue_function` fills `constant_memory` mappings through the function properties `global_address:NAME` and `global_size:NAME` instead of refusing them                                                   | modified                                                                       |
| `max/mojo/max/_driver/max_license/hal/context.mojo`                             | `get_optional_function_property` on a context                                                                                                                                                               | modified                                                                       |
| `max/mojo/max/_driver/max_license/hal/plugin.mojo`                              | `get_optional_function_property` on the raw driver                                                                                                                                                          | modified                                                                       |
| `max/mojo/max/_driver/max_license/hal/plugins/base.mojo`                        | binds `M_driver_function_property` for the above                                                                                                                                                            | modified                                                                       |
| `max/mojo/max/_driver/max_license/hal/_plugin/intel/__init__.mojo`              | Intel HAL plugin package                                                                                                                                                                                    | new, written for this work after the existing vendor plugins in `hal/_plugin/` |
| `max/mojo/max/_driver/max_license/hal/_plugin/intel/intel_plugin.mojo`          | Intel HAL plugin hooks: execution configuration and the Intel extension set, named by the Intel MAX plugin's `HalHooks`                                                                                     | new, as above                                                                  |
| `max/mojo/max/_driver/max_license/hal/extensions/intel.mojo`                    | Intel HAL extension: device features (`M_driver_intel_device_features`)                                                                                                                                     | new, as above                                                                  |

The new files carry the MAX License header, with Modular's copyright line, that
their neighbours carry. No file outside these directories was derived from
MAX-licensed code.
