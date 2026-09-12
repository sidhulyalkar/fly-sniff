# Task optimization v1: make the model better without training away the connectome

`fly-sniff` now has a positive preregistered E001 structural result, but structure alone does not determine neural dynamics. The next question is therefore not "can we hand-tune the graph until it navigates?" It is:

> **If MaleCNS supplies the fixed wiring scaffold, can a very small set of unknown dynamical parameters be task-optimized to support odor-source navigation, and does that scaffold remain useful when the same optimization budget is given to degree-preserving rewired controls?**

This follows the logic of connectome-constrained, task-optimized mechanistic models: measured connectivity is treated as structural authority while unknown dynamical parameters are fit to a computational task. It does **not** claim that a living fly is being trained or that the optimized parameters are measured physiology.

## What is fixed

For every optimization run:

- exact graph body IDs and directed edges are fixed;
- structural synapse counts and transmitter-derived signs are fixed;
- role membership is fixed before training;
- the plume, sensor model, and runtime observation interface remain explicit;
- the controller never receives source coordinates, distance-to-source, or audience-only plume ground truth.

The optimizer may use distance-to-source to compute a reward during development. That is a training signal, analogous to supervised/reinforcement feedback, not an inference-time sensory input.

## Eight trainable degrees of freedom

`configs/task_optimization_v1.json` freezes only eight global parameters:

1. `tau_s`: modeled neural relaxation time constant;
2. `activation_gain`: global nonlinear state gain;
3. `recurrent_gain`: global scale on recurrent connectome drive;
4. `odor_gain`: one bilateral gain shared by left and right modeled odor drives;
5. `wind_forward_gain`: positive body-forward airflow gain;
6. `wind_backward_gain`: negative body-forward airflow gain;
7. `wind_cross_gain`: one gain tied across left and right crosswind drives;
8. `turn_gain`: bilateral steering-readout gain.

The optimization is deliberately low-dimensional. Individual edges are **not** free parameters. If a model with eight global dynamical parameters cannot exploit the structural scaffold, adding thousands of edge-specific weights would make the result less informative rather than more convincing.

## Development objective

The frozen development objective is

`0.50 * success + 0.40 * SPL + 0.10 * terminal_progress`.

`terminal_progress` is the clipped normalized change in distance to the goal region. It exists only to provide signal before the controller achieves its first successes. Final reporting still uses success rate, SPL, path statistics, and matched uncertainty rather than the training reward.

## Hard train/final split

Training and development validation use seeds from the integer namespace starting at `2,100,000,000`. The sealed final benchmark samples only from `1..1,999,999,999`. This creates a non-overlapping seed split by construction.

The final held-out and OOD cohorts must not be used for:

- optimizer updates;
- model selection;
- choosing parameter bounds;
- deciding which training algorithm looks best;
- curriculum tuning;
- selecting a favorable rewire seed.

Once final evaluation is opened, it is reporting, not development.

## Matched topology controls are mandatory

Training the intact connectome and comparing it against an untrained rewire would confound topology with optimization. Therefore every topology receives the exact same:

- training/development seed sets;
- objective;
- parameterization and bounds;
- optimizer seed;
- population size and generations;
- candidate episode budget;
- stopping rule.

The development cohort is:

1. intact reviewed MaleCNS-derived graph;
2. eight separately trained directed degree-preserving rewires;
3. the prespecified steering-input lesion, also separately trained.

The primary development question is whether separately optimized intact wiring exceeds the mean separately optimized rewire ensemble on untouched development-validation seeds. The lesion asks whether the trained controller still depends on the frozen steering route.

## Optimizer

v1 uses a deterministic log-space cross-entropy method (CEM). This is intentional:

- the simulator contains nondifferentiable movement/success events;
- the parameter space is only eight-dimensional;
- all candidates in a generation are evaluated on the same environment seeds (common random numbers);
- optimization can remain NumPy/SciPy-only rather than introducing a second neural framework into the scientific core.

The optimized artifact must record every bound, optimizer setting, generation seed subset, graph fingerprint, and final parameter value.

## Qualification order

Task optimization does not bypass anatomical or dynamical qualification. The intended order is:

1. **E001:** audited body-ID-continuous structural route;
2. **human route review:** laterality, intermediate populations, signs, and role assignments;
3. **candidate GraphBundle:** exact reviewed nodes/edges/roles;
4. **task optimization:** development seeds only;
5. **E002 on the trained parameter set:** mirrored inputs, correct steering laterality, deterministic replay, lesion dependency, sign coverage;
6. **matched trained-control development comparison:** intact vs eight trained rewires vs trained lesion;
7. **freeze code + parameters + final manifest**;
8. **one-way held-out/OOD evaluation**.

A trained model that navigates well but fails E002 is not acceptable for the public connectome claim.

## What a positive result would mean

A positive final result could support:

> **A task-optimized dynamical model constrained by the audited MaleCNS wiring navigated the plume better than equally task-optimized degree-preserving rewired controls under the preregistered benchmark.**

That is stronger and more precise than "we trained a fly brain." It tests whether the measured topology contributes useful structure after both the intact network and null topologies receive a fair chance to adapt their unknown dynamics.

## Biological plasticity is a separate second lane

Drosophila does possess biologically grounded olfactory learning mechanisms, especially dopamine-gated plasticity in mushroom-body circuits. We should test those next, but not conflate them with task optimization.

A later `plasticity-v1` experiment should restrict learning to literature-supported synaptic classes such as odor-active KC→MBON pathways and explicit DAN-gated updates. That experiment asks whether a plausible local learning rule can adapt behavior. The present experiment asks the cleaner topology question first.
