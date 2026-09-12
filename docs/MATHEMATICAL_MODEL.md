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

Arena boundaries reflect heading after position is clipped back to the boundary. This is benchmark geometry, not a wall-interaction model.

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

At each antenna, concentration \(c\) is first scaled and saturates through

\[
r = gc,
\qquad
s=\frac{r}{K+r},
\]

where \(g\) is `concentration_gain` and \(K\) is `concentration_half_sat`.

Adaptation is modeled as a first-order state

\[
\tau\frac{da}{dt}=s-a.
\]

For piecewise-constant drive during one simulator step, the implementation uses the exact zero-order-hold update

\[
a_{t+\Delta t}=a_t+\left(1-e^{-\Delta t/\tau}\right)(s-a_t).
\]

The reported antenna signal is

\[
y=\operatorname{clip}_{[0,1]}
\left(0.72s+0.28\max(s-a,0)\right).
\]

This final mixture is an explicit **phenomenological benchmark transduction**, not a receptor-kinetics model and not measured ORN firing.

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

Graph distance, degree, and structural edge weight are **not neural activity, functional importance, or causal influence**.

## 8. Evidence classes for visualization

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

## 9. Determinism and paired comparisons

For a fixed episode seed, compared controllers receive the same exogenous plume realization. Recordings are canonicalized and SHA-256 hashed. Rendering is downstream of the scientific recording and must not alter controller state, plume state, success, or trajectory.

## 10. Current claim boundary

Until the repository qualification gates pass, public development artifacts must remain labeled:

`DEVELOPMENT PROXY • NOT A MALECNS RESULT`

A structural corridor is a discovery result. Promotion to a connectome-controller claim additionally requires sealed body IDs/roles, sign provenance where available, bilateral perturbation checks, steering-output lesion checks, deterministic replay, and the preregistered behavioral evaluation.
