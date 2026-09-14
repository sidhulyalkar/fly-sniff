# Project scientific programs v1

The repository contains several legitimate but scientifically different kinds of experiment. They must not
share a headline claim merely because they use the same MaleCNS graph.

## Program A — latent wiring

Question:

> How much task-relevant computation is already latent in biological wiring before task-specific learning?

Rules:

- calibrate only against independent physiology/circuit constraints;
- never use navigation reward during calibration;
- freeze one dynamics parameterization before the headline behavioral benchmark;
- apply the same frozen dynamics to intact and null topologies;
- use acute lesions to test causal dependence;
- treat topology as the experimental unit for topology claims;
- preserve negative results.

This is the primary future odor-navigation claim lane.

## Program B — topology as inductive bias

Question:

> Does MaleCNS topology learn a task better or faster than matched null graphs?

Rules:

- task optimization is allowed;
- every topology receives the same parameterization freedom, seeds, optimizer, and compute budget;
- a trained intact graph is never compared with an untrained null as topology evidence;
- output must be described as a connectome-constrained learned model, not a latent innate policy.

The existing sealed task-optimization lane belongs here.

## Program C — biological learning

Question:

> Can a prespecified biological plasticity mechanism modify behavior while the rest of the circuit remains
> constrained?

Rules:

- plasticity locations and learning rules are selected from independent biological evidence;
- whole-connectome backpropagation is not permitted in v1;
- the experiment reports what changed and what remained frozen;
- reward can influence the specified biological learning mechanism without becoming a global optimizer.

A future mushroom-body KC→MBON learning experiment is a natural first target.

## What the project is not claiming

The project does not treat the MaleCNS connectome as a complete executable fly. The connectome constrains
structure. Dynamics, sensory transduction, neuromodulation, receptor effects, plasticity, biomechanics, and
environmental coupling require additional evidence or explicit model assumptions.

The project therefore asks not simply whether a connectome-shaped network can perform a behavior, but which
parts of the behavior depend on the biological wiring, which depend on added dynamics, and which depend on
learning.

## Branch ownership

See GitHub issue #24 for the live integration map. Scientific branches should identify their program and
state which assumptions they are forbidden to change before any performance-bearing result is inspected.
