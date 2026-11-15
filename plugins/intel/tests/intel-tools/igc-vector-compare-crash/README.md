# IGC: the vectorizer crashes on an `extractelement` of a vector compare

`specials_atan2.spv` (17 KB) is a module the Mojo compiler plugin built for
a kernel that computes `atan2` on four lanes, every lane loaded from
memory. It passes `spirv-val`. `run.sh` builds it for `bmg-g31` with
`ocloc`: with the driver plugin's options, with `-cl-opt-disable` added,
and with IGC's vectorizer turned off (`IGC_EnableVectorizer=0`).

```sh
sh run.sh
```

It needs `spirv-val` and `ocloc`; no GPU.

## Expected and actual

The module should compile with every option set. With IGC 2.41.5:

```text
spirv-val: valid
ocloc -options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt": exit 245
[bmg-g31] IGC: Internal Compiler Error: Floating point exception
ocloc -options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt -cl-opt-disable": exit 0
IGC_EnableVectorizer=0 ocloc -options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt": exit 0
```

| IGC                                      | Default options                                                                | `-cl-opt-disable` | `IGC_EnableVectorizer=0` |
|------------------------------------------|--------------------------------------------------------------------------------|-------------------|--------------------------|
| 2.41.5                                   | exit 245, floating point exception                                             | exit 0            | exit 0                   |
| v2.42.14                                 | exit 245                                                                       | exit 0            | exit 0                   |
| master `bd0ad0a5`, built with assertions | exit 134: assertion `eltBytes` in `EmitPass::emitExtract` (`EmitVISAPass.cpp`) | exit 0            | exit 0                   |
| v2.43.9                                  | exit 245                                                                       | exit 0            | exit 0                   |
| master `bd0ad0a5` with `fix.patch`       | exit 0                                                                         | exit 0            | -                        |

## Cause and patch

After making a vector `icmp`, `IGCVectorizer::handleCMPInstruction`
(`IGC/Compiler/CISACodeGen/IGCVectorizer.cpp`) replaces every use of the
scalar compares with an `extractelement` of the `<4 x i1>` result.
Selects it vectorized use the vector compare directly, but selects that
stay scalar get the `extractelement`, which `EmitPass::emitExtract` cannot
emit: an `i1` lane has no whole bytes, so `eltBytes` is 0. An assertion
build stops there; a release build divides by it (`SIGFPE`, exit 245).

`fix.patch` keeps the scalar compares for their scalar users and maps them to
the vector compare for the vectorizer, so no `i1` vector is ever extracted;
compares with no other users die. It is against IGC master `bd0ad0a5` (MIT, like
IGC). With it the module compiles in a release and in an assertion build of
`bd0ad0a5`.
