# FlyBrain Plume Hunt benchmark contract

This document defines success **before** final evaluation.

## Scientific question

> Does biological MaleCNS topology provide a measurable inductive advantage for odor-source navigation under intermittent turbulent sensory evidence?

The benchmark does not ask whether a hand-engineered mapping can make a fly connectome move. It asks whether a reviewed connectome-derived circuit performs differently from matched topology controls under identical sensory input.

## Primary endpoint

**Success weighted by path length (SPL).** For episode `i`:

`SPL_i = success_i * shortest_path_i / max(shortest_path_i, actual_path_i)`

Failed episodes have SPL 0.

## Gold criterion

A result qualifies as **FlyNav Gold** only if all conditions hold on the sealed final test:

1. MaleCNS source-finding success >= **0.70** across **1,000** held-out in-distribution plume episodes.
2. `mean SPL(MaleCNS) - mean SPL(degree-preserving rewire) >= 0.10`.
3. The paired bootstrap 95% CI of that SPL difference excludes 0.
4. MaleCNS source-finding success >= **0.60** across **400** sealed OOD plume episodes.
5. No final-test seed, plume parameter, role mapping, dynamics parameter, or stopping rule changed after unblinding.

## Required controls

- degree-preserving directed rewiring of the extracted graph;
- a transparent classical cast-and-surge controller;
- a capacity-appropriate artificial recurrent baseline in the next tranche;
- optional lesion/role ablations after the primary comparison.

All controllers receive identical observations and paired episode seeds. Plumes are exogenous: an agent cannot alter the plume realization.

## What is trainable?

The first topology test should minimize free parameters. Global gain/leak/transduction constants may be calibrated on development seeds. Individual connectome edge weights may **not** be reward-trained for the zero/few-shot topology claim.

If zero/few-shot topology fails, sample efficiency becomes a secondary experiment. At that point all trainable parameters and optimization budgets must be matched across topology controls.

## Anti-cherry-picking rules

- Development videos are watermarked `DEVELOPMENT`.
- The social renderer never invents benchmark statistics.
- Final figures must be generated from a receipt whose manifest hash matches the sealed test manifest.
- A visually attractive episode may illustrate behavior, but headline metrics come from the complete held-out cohort.
- Negative results are retained and reported.

## Staged gates

- **E001 Data authority:** exact upstream resource + reproducible extraction.
- **E002 Circuit sanity:** sensory injection propagates and left/right steering outputs are measurable.
- **E003 Easy plume:** qualified graph beats random walk in smooth/intermittent conditions.
- **E004 Turbulent plume:** meaningful source localization under stochastic puffs.
- **E005 Nulls:** graph vs degree-preserving rewires under paired seeds.
- **E006 Freeze:** seal role map, dynamics version, seeds, and plume distributions.
- **E007 Final:** run once, issue immutable receipt, then render.
