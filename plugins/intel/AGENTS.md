> [!NOTE]
> This file was written by an AI agent for AI agents. Check every command,
> path and version in it against the current sources before relying on it.

# Agent guidelines for the Intel GPU plugins

If AI is heavily used for any users interested in this fork, may it at
least follow some basic guidelines.

Instructions for coding agents working on Intel GPU support in this tree:
`plugins/intel/` and the changes elsewhere that it needs. The repository's
top-level `AGENTS.md` and Modular's `AI_TOOL_POLICY.md` apply first.

## Skills

Task-specific instructions (building and running on Intel GPUs, the checks,
the codegen and driver plugins, Intel kernels, performance, quantization,
bugs in Intel's tools, building and bisecting IGC, GPU fault recovery,
diagnostic bundles, and opening issues and pull requests) are kept as agent
skills in
[team-blue-savery42/skills](https://github.com/team-blue-savery42/skills).
Start with its `AGENTS.md`.

## Rules that apply here

- Read `plugins/intel/README.md`, `plugins/intel/CONTRIBUTING.md` and the
  sources beside your change before writing code; the plugin interfaces
  change often.
- State a result on an Intel GPU only if you ran it, with the GPU, driver
  stack versions and command. `plugins/intel/utils/check` is the check to
  run before a series is sent.
- Known bugs in Intel's tools, their versions and workarounds are in
  `plugins/intel/docs/intel-tools-issues.md`. IGC 2.42 and 2.43 return NaN
  from MAX's paged FA; the tested IGC is 2.41.5.
- Label AI-assisted work as `AI_TOOL_POLICY.md` asks, and keep a human in
  the loop: do not open issues or pull requests, or comment, without the
  contributor's review.
- Every issue, pull request, comment or upstream report an agent opens or drafts
  starts with this block, exactly:

  ```markdown
  > [!NOTE]
  > I am an unmanaged AI agent and this has not seen human review, feel free to
  > close and ignore.
  ```

  Never remove or reword it yourself. Only the human author removes it, after
  reviewing the text; AI-generated code is fine, and that review before
  submission is where the note is caught.
