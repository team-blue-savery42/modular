# Contributing to the Intel GPU plugins

These rules apply to every change under `plugins/intel/` and to the
changes elsewhere in the tree that the Intel plugins need.

## Commits

- One logical change per commit. Every commit builds and passes its
  tests, so the series bisects. A fix to an earlier commit of the same
  series is squashed into that commit.
- Title: `module: short description`, imperative mood, at most 72
  characters. Modules in use: `intel/compiler`, `intel/driver`,
  `intel/utils`, `intel/tests`, `intel/docs`, `bazel`, `compiler`,
  `stdlib`, `max`, `kernels`, `mlrt`, `pipelines`.
- Body: wrapped at 72 columns, plain technical language. State the
  problem, its cause and the change. Quote measurements where they
  justify the change: hardware, command, before and after. No narrative
  and no history of the experiments that led to the change.
- Every commit carries `Signed-off-by:` (Developer Certificate of
  Origin, `git commit -s`). A commit with substantial AI-generated
  content also carries `Assisted-by: AI` before it, as Modular's
  `AI_TOOL_POLICY.md` asks. No other trailers.
- Commit messages, comments, test names and file names describe the
  code and the hardware, never external planning documents or issue
  trackers that are not public.

## Code style

The series targets Modular's repository, so Modular's written guides and
linters take precedence; the LLVM Coding Standards apply where they say
nothing.

Modular's written guidance:

- `CONTRIBUTING.md`: format with `./bazelw run //:format` (clang-format,
  buildifier, mblack for Mojo, ruff for Python, rumdl for Markdown,
  shellcheck, and the license-header check).
- `Mojo/docs/contributing/compiler/README.md`: no C++ style guide;
  match the conventions of the surrounding code and run the formatter.
- `Mojo/docs/contributing/stdlib/stdlib-code-style.md` and
  `docstring-style-guide.md`: Mojo naming (snake_case functions and
  variables, PascalCase types and type parameters), lexicographically
  sorted explicit imports, docstrings, the license header, and
  target-specific code gated on an explicit check of the target (no
  `else` that falls through to one vendor).
- `.clang-format` (LLVM based) and `.clang-tidy`
  (`readability-identifier-naming`, warnings as errors).

Where Modular and LLVM differ:

| Topic                      | LLVM Coding Standards                       | Modular                                                                                             | Used here |
|----------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------|
| Local variables            | CamelCase                                   | camelBack (`.clang-tidy`, MLIR style; the HIP and CPU plugins)                                      | camelBack |
| Members                    | CamelCase                                   | camelBack in the HAL plugins; trailing `_` for private members in MLRT                              | camelBack |
| Functions                  | camelBack                                   | camelBack                                                                                           | camelBack |
| Types, namespaces          | CamelCase                                   | CamelCase                                                                                           | CamelCase |
| File header                | banner, file name, description              | banner with the copyright and license, then the description                                         | Modular   |
| Errors                     | `llvm::Error` and `Expected`, no exceptions | `M_driver_status_t` at the HAL C ABI, with the message on the driver; LLVM's types in compiler code | Modular   |
| `auto`, includes, comments | as written                                  | the same, through clang-format and the existing code                                                | LLVM      |

Rules:

- C++ is formatted with the repository's `.clang-format` and is clean
  under its `.clang-tidy`.
- Mojo and Python follow the stdlib guide and the files beside them, and
  `//:format` leaves them unchanged.
- Every new file carries Modular's license header for its directory:
  the Apache-2.0 with LLVM exceptions header, or the MAX License header
  under `max_license/`, `MLRT/` and `GraphCompiler/`. Its copyright line
  names the file's author, and Modular too where the file follows one of
  Modular's; changed upstream files keep Modular's line.
  `docs/LICENSING.md` lists the MAX-licensed files this work changes.

## Checks

Run `plugins/intel/utils/check {{ OUTDIR }}` on a host with an Intel GPU before
sending a series: it builds everything, runs the plugin unit tests, the
codegen corpus, the E2E probes, the MAX graph tests, MAX's GPU
test gate against `tests/max-gpu-expected.txt`, and compares
Qwen3.5-0.8B's greedy text with `tests/regression/`. A change that moves
a gate result updates `tests/max-gpu-expected.txt` with the cause.
