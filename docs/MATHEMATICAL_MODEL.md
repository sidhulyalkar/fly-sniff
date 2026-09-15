# Mathematical model contract

This document states the equations implemented by the development benchmark. It is a reproducibility contract, not a claim that every phenomenological term is a faithful model of Drosophila physiology or room-scale fluid dynamics.

## 1. Arena and kinematics

The agent state is position \((x_t,y_t)\) and heading \(\theta_t\). A normalized steering command \(u_t\in[-1,1]\) is converted to angular velocity with the configured maximum turn rate \(\omega_{\max}\):

\[
\theta_{t+\Delta t}=\mathrm{wrap}\left(\theta_t + u_t\,\omega_{\max}\,\Delta t\right).
\]

With speed scale \(s_t\), configured forward speed \(v\), and clipping to the supported speed range, translation is

\[
\begin{aligned}
x_{t+\Delta t} &= x_t + v s_t\cos(\theta_{t+\Delta t})\Delta t,\\
y_{t+\Delta t} &= y_t + v s_t\sin(\theta_{t+\Delta t})\Delta t.
\end{aligned}
\]

The coordinate convention is explicit: **positive turn is left/counterclockwise** and negative turn is right/clockwise. Arena boundaries reflect heading after position is clipped back to the boundary. This is benchmark geometry, not a wall-interaction model.

## 2. Stochastic puff plume

Each odor puff has center \((x_i,y_i)\), age \(a_i\), Gaussian width \(\sigma_i\), and mass \(m_i\).

### Emission

During one simulation step, the number of new puffs is

\[
N_t\sim\operatorname{Poisson}(\lambda\Delta t),
\]

where \(\lambda\) is `emission_rate_hz`. New puffs are emitted at the configured source x-position with a small Gaussian y-jitter.

### Center transport

The world-frame mean wind is constant and points downwind along +x:

\[
\Delta x_i = U\Delta t.
\]

Crosswind center motion combines a deterministic meander term and Brownian-like stochastic displacement:

\[
\Delta y_i = A\sin(ft + 0.19a_i)\Delta t + \sigma_c\sqrt{\Delta t}\,\xi_i,
\qquad \xi_i\sim\mathcal N(0,1).
\]

The stochastic center displacement and within-puff diffusion are deliberately separate benchmark mechanisms.

### Puff diffusion

Puff width obeys

\[
\sigma_i^2(a_i)=\sigma_0^2+2Da_i,
\]

where \(D\) is `diffusion_rate`.

### Concentration field

The concentration contribution of each puff is a normalized isotropic 2-D Gaussian:

\[
c_i(x,y)=\frac{m_i}{2\pi\sigma_i^2}
\exp\left[-\frac{(x-x_i)^2+(y-y_i)^2}{2\sigma_i^2}\right].
\]

The field is the sum over retained puffs. For computational speed, contributions farther than \(4\sigma_i\) in either coordinate are omitted. This truncation is part of the benchmark implementation.

This is a 2-D stochastic puff benchmark. It is **not CFD**, does not model obstacle-aware flow, and must not be visualized as though furniture or people deflect the plume.

## 3. Bilateral antenna geometry

For antenna separation \(d\), left and right samples are placed at \(\pm d/2\) along the body lateral axis. For body heading \(\theta\):

\[
\begin{aligned}
p_L &= (x,y)+\frac d2(-\sin\theta,\cos\theta),\\
p_R &= (x,y)-\frac d2(-\sin\theta,\cos\theta).
\end{aligned}
\]

Thus at heading zero (+x), the left antenna is at +y and the right antenna is at -y.

## 4. Phenomenological odor transduction

At each antenna, concentration \(c_t\) is first scaled and saturates through

\[
r_t = g c_t,
\qquad
s_t=\frac{r_t}{K+r_t},
\]

where \(g\) is `concentration_gain` and \(K\) is `concentration_half_sat`.

Adaptation is modeled as a first-order state

\[
\tau\frac{da}{dt}=s-a.
\]

The observation at time \(t\) is calculated from the current saturated drive and the adaptation state **carried into** that sample:

\[
y_t=\operatorname{clip}_{[0,1]}
\left(0.72s_t+0.28\max(s_t-a_t,0)\right).
\]

Only after emitting that observation is the adaptation state advanced. For piecewise-constant drive during one simulator step, the implementation uses the exact zero-order-hold update

\[
a_{t+\Delta t}=a_t+\left(1-e^{-\Delta t/\tau}\right)(s_t-a_t).
\]

This ordering avoids using a future-updated adaptation state in the current sensory observation. The final mixture is an explicit **phenomenological benchmark transduction**, not a receptor-kinetics model and not measured ORN firing.

### Observation idempotence

Reading an observation more than once at the same physical simulator state must not advance adaptation. Adaptation is stateful and may advance only when physical simulator state/time advances. Logging and rendering therefore cannot change behavior by calling `observe()` repeatedly.

## 5. Airflow in the fly body frame

The world-frame wind vector \(w\) is rotated into body coordinates by \(-\theta\):

\[
w_{body}=R(-\theta)w.
\]

Controllers receive only this local airflow vector. They are not given source coordinates or a privileged world-frame source bearing.

## 6. Development controller

The `BilateralProxyController` is intentionally transparent and is **not** a MaleCNS model.

It uses:

- mean bilateral odor as a gate for odor-contact versus odor-lost behavior;
- body-frame airflow to orient upwind or crosswind;
- right-minus-left odor difference as a small bilateral steering bias;
- a simple exponentially weighted odor memory.

A stronger right antenna signal produces a negative/rightward correction and a stronger left antenna signal produces a positive/leftward correction. Its `dn_left` and `dn_right` diagnostics are display-only proxy motor channels aligned with that convention. They are not descending-neuron recordings.

All constants in this controller are engineering parameters for a development baseline. They must not be described as fitted neural physiology.

## 7. Structural MaleCNS tracing

The structural tracer performs bounded graph discovery between source and target seed sets.

For each direction it:

1. removes edges below `min_weight`;
2. expands at most `fanout_per_node` strongest local edges per frontier node;
3. records minimum discovered graph depth up to `max_hops`.

A retained node must be able to participate in a bounded source-to-target corridor under the discovered forward and reverse depth maps. Persisted edges must:

- connect retained nodes;
- satisfy `weight >= min_weight`;
- satisfy the bounded source-to-target depth criterion.

The report distinguishes regex-matched **input seeds** from seeds that actually survive into the bounded corridor. Graph distance, degree, and structural edge weight are **not neural activity, functional importance, or causal influence**.

## 8. Explicit MaleCNS rate-model assumption

`MaleCNSRateController` is a modeled dynamical system laid over a reviewed structural graph. It is not a reconstruction of membrane voltage, spike timing, calcium activity, or measured physiology.

### Signed structural matrix

For a directed structural edge from presynaptic neuron \(j\) to postsynaptic neuron \(i\), the raw modeled coefficient is

\[
q_{ij}=\operatorname{sign}_{ij}\log(1+w_{ij}),
\]

where \(w_{ij}>0\) is the measured structural synapse-count weight and `sign` is the explicit presynaptic sign policy. Sign 0 means unresolved and therefore contributes zero drive rather than being silently treated as excitatory or inhibitory.

For numerical stability, incoming coefficients are normalized per postsynaptic neuron:

\[
W_{ij}=\frac{q_{ij}}{\sum_k |q_{ik}|}
\]

when the denominator is nonzero. This normalization preserves relative signed input structure within a postsynaptic row but **does not preserve absolute synaptic drive across neurons**. It is an engineering modeling choice and must not be described as physiological synaptic strength.

### Modeled dynamics

Given modeled activity vector \(a_t\) and external role drive \(u_t\), define

\[
p_t=\tanh\left(g(Wa_t+u_t)\right).
\]

The state is treated as a first-order relaxation toward this nonlinear proposal:

\[
\tau_a\frac{da}{dt}=p-a.
\]

For one controller step, the implementation uses

\[
\rho=e^{-\Delta t/\tau_a},
\qquad
a_{t+\Delta t}=\rho a_t+(1-\rho)p_t.
\]

The default \(\tau_a\) was chosen to reproduce the historical v0 per-step retention of 0.82 at \(\Delta t=0.05\,\mathrm{s}\). It is therefore a compatibility-preserving engineering parameter, **not a fitted neural time constant**. The explicit time-constant form ensures that the same constant-input relaxation is consistent when simulation `dt` changes.

### Sensory injection

The benchmark adapter injects recorded behavioral variables into named roles:

- `odor_left`, `odor_right` receive the corresponding modeled antenna signal;
- `wind_forward`, `wind_backward`, `wind_left`, `wind_right` receive rectified components of body-frame airflow.

This mapping is an engineered interface between the plume benchmark and reviewed body-ID roles. It is not evidence that each named neuron receives those variables with unit gain in vivo.

### Steering readout and laterality

Let \(L\) and \(R\) be the mean modeled activities of `steer_left` and `steer_right`. The current steering readout is

\[
u_t=\tanh\left(k_{turn}(L-R)\right).
\]

Because FlySniff defines positive turn as left/counterclockwise, `steer_left` is required to drive positive/left turning and `steer_right` negative/right turning. Qualification now checks this laterality explicitly rather than accepting merely opposite signs.

This ipsilateral convention is consistent with published PFL3 steering results, including:

- Hulse et al., *eLife* (2021), doi:10.7554/eLife.66039;
- Westeinde et al., *Nature* (2024), doi:10.1038/s41586-023-07006-3;
- Matheson et al., *Nature Communications* (2022), doi:10.1038/s41467-022-32247-7.

Those papers are biological priors and sign-convention references. They do not by themselves assign exact MaleCNS v1.0 body IDs to this project's roles.

### Current scope

The v0 connectome controller returns fixed normalized forward speed and uses the qualified graph to model steering. It must therefore **not** be described as a complete model of PFL2-mediated speed control or full fly locomotor circuitry. Adding connectome-derived speed modulation requires a separate qualified role/readout and ablation.

## 9. Evidence classes for visualization

The renderer must keep these classes distinct:

### Measured structure

- exact MaleCNS body IDs;
- exact directed structural edges;
- structural edge weights;
- annotation and sign provenance when available.

### Modeled state

- explicit controller/rate-model states generated by the codebase;
- never described as recorded physiology.

### Behavioral state

- antenna signals;
- body-frame airflow;
- turn/speed commands;
- position, heading, and source-finding state.

A structural graph may glow as `MODELED ACTIVITY` only when an explicit model generated that activity and the recording stores it. Structural connectivity alone must never be animated or labeled as firing.

## 10. Determinism and paired comparisons

For a fixed episode seed, compared controllers receive the same exogenous plume realization. Recordings are canonicalized and SHA-256 hashed. Rendering is downstream of the scientific recording and must not alter controller state, plume state, success, or trajectory.

## 11. Current claim boundary

Until the repository qualification gates pass, public development artifacts must remain labeled:

`DEVELOPMENT PROXY • NOT A MALECNS RESULT`

A structural corridor is a discovery result. Promotion to a connectome-controller claim additionally requires sealed body IDs/roles, sign provenance where available, correct bilateral perturbation laterality, steering-output lesion checks, deterministic replay, and the preregistered behavioral evaluation.
