load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Level Zero API headers (MIT), included as <level_zero/ze_api.h>.
cc_library(
    name = "headers",
    hdrs = glob(["include/*.h"]),
    include_prefix = "level_zero",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
