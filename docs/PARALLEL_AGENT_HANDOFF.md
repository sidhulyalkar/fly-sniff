# Parallel-agent handoff

All agents working on `fly-sniff` should treat GitHub issue #24 as the coordination source of truth.

Before starting a branch:

1. identify the scientific program: latent wiring, topology inductive bias, or biological learning;
2. identify the work lane from issue #24;
3. state what the branch is forbidden to change;
4. inspect existing open PRs for overlapping ownership;
5. prefer exporting artifacts into the shared experiment/evidence kernel rather than importing another lane's
   performance logic.

Current boundaries:

- PR #22 owns experimental-plume source validation and must not grant controller access;
- PR #23 owns the structural olfactory-motion candidate audit and must not use navigation performance;
- the PR #20 task-optimization lineage is Program B and its final namespace should remain unconsumed while
  Program A is being defined;
- `feat/experiment-evidence-kernel-v1` owns generic evidence/spec/lock/receipt infrastructure only;
- visualization should replay sealed outputs rather than implementing independent neural dynamics.

If two branches encode conflicting scientific assumptions, preserve both branches and resolve the protocol
choice in issue #24 before looking at behavior. Do not select whichever assumption produces the prettier or
higher-scoring result.
