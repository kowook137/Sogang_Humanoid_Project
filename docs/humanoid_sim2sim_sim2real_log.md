## Summary

This PR builds a Berkeley Humanoid Lite MuJoCo sim2sim workflow that is usable from WSL and extends it into a Sim2Real-oriented walking baseline.

The work started from repository/runtime integration and keyboard control, then progressed through low-speed gait startup, policy behavior analysis, heading/path stabilization, discrete walking-speed control, foot-clearance measurement, and lateral path correction.

> **Last updated: 2026-10-02**

---

## Engineering Timeline

### 2026-08-19 — Environment and simulator bring-up

**Environment verified**
- Python 3.11
- PyTorch 2.7.0 + CUDA 12.8
- Isaac Sim 5.1.0
- Isaac Lab 2.3.2
- MuJoCo 3.3.5
- CUDA execution confirmed in the WSL development environment.

**Problem**
- The team repository did not have the Berkeley Humanoid Lite assets and low-level repositories registered as root-level submodules.
- The released MJCF paths did not match the local repository mesh layout.
- The original simulator expected gamepad input, which was inconvenient in the WSL development environment.

**Changes**
- Registered the Berkeley Humanoid Lite Assets and Lowlevel repositories as pinned submodules.
- Updated MuJoCo asset loading for the released repository layout.
- Resolved the MJCF mesh-path mismatch at runtime using temporary corrected XML files without modifying the assets submodule.
- Added support for the full 22-joint humanoid model with arms.
- Replaced the gamepad controller with persistent keyboard SE(2) commands:
  - `W / S`: forward / backward
  - `A / D`: lateral movement
  - `Q / E`: yaw
  - `X` or `Space`: stop
  - `H`: help
- Modified the control loop so that it exits when the MuJoCo viewer closes.

**Result**
- Full 22-joint humanoid model launched successfully in MuJoCo.
- Keyboard forward and stop commands were successfully verified.

**Commit**
- `83a313d Add keyboard-controlled humanoid sim2sim`

---

### 2026-08-19 — WSLg D3D12 rendering acceleration

**Problem**
- MuJoCo under WSLg could fall back to software rendering, reducing simulator responsiveness.

**Changes**
- Added WSL environment detection.
- Automatically selected Mesa's D3D12 backend before MuJoCo creates its OpenGL context when `/dev/dxg` and the D3D12 driver are available.
- Added automatic `GALLIUM_DRIVER=d3d12` configuration.

**Result**
- WSLg D3D12 hardware acceleration was confirmed.
- MuJoCo rendering responsiveness improved.

**Commit**
- `23be6a8 Enable WSLg D3D12 rendering for MuJoCo`

---

### 2026-08-19 — Policy comparison and low-speed walk-start boost

**Policy analysis**
- The full-body 22-action policy showed poor low-speed forward locomotion near `vx=0.30 m/s`.
- Increasing the command produced unstable acceleration and significant upper-body action/torque behavior.
- The 12-action leg locomotion policy showed substantially cleaner walking behavior.
- The leg policy could sustain approximately `vx=0.30 m/s` once walking had started, but required roughly `vx=0.40 m/s` to leave the standing attractor from rest.

**Changes**
- Added a temporary walk-start boost for the 12-action leg policy:
  - walking threshold: `0.30 m/s`
  - startup policy velocity: `0.40 m/s`
  - duration: `1.0 s`

**Result**
- The robot could initiate walking from rest and then return to the requested low-speed command.

**Commit**
- `c3fc071 Add low-speed walk-start boost`

---

### 2026-08-31 — Low-speed gait and path-tracking development

**Goal**
- Improve low-speed walking behavior and reduce repeated heading drift.

**Changes**
- Added smoothed forward-command handling.
- Evaluated the cleaner `vx=0.50 m/s` gait as an initial strategy for low-speed walking.
- Added heading hold based on the captured straight-ahead yaw.
- Added line-based cross-track tracking.
- Added path-heading correction:
  - path-heading gain: `1.2`
  - heading-offset limit: `12 deg`
- Added detailed gait diagnostics:
  - requested velocity
  - policy velocity
  - world velocity
  - body-frame forward velocity
  - cross-track error
  - path-heading correction
  - yaw
  - yaw error
  - yaw command
  - yaw rate
  - passive-arm torque RMS
  - passive-arm saturation
  - base angular speed

**Result**
- Walking and path-tracking behavior became quantitatively measurable instead of relying only on visual inspection.
- The diagnostic framework became the basis for subsequent Sim2Real-oriented gait tuning.

**Commit**
- `0e2f7c2 Improve low-speed gait and path tracking`

---

### 2026-10-02 — Walking-speed characterization

**Goal**
- Determine a practical nominal walking speed for eventual physical-humanoid transfer instead of forcing the learned policy into an unnatural low-speed gait.

**Changes**
- Added discrete forward/backward walking-speed levels:
  - `0.30 m/s`
  - `0.40 m/s`
  - `0.45 m/s`
  - `0.50 m/s`
- Opposite-direction input now reduces the command toward zero before changing direction.

**Observed gait behavior**
- `0.30 m/s`
  - Short stepping motion.
  - Low foot clearance.
  - Policy could sustain the gait after startup but walking quality was relatively poor.
- `0.40 m/s`
  - Clear improvement in stepping motion and foot clearance.
- `0.45 m/s`
  - Clear stepping gait while remaining slower than the policy's strongest gait.
  - Selected as the current nominal Sim2Real-oriented walking speed.
- `0.50 m/s`
  - Produced the largest and cleanest gait in MuJoCo.
  - Retained as an upper/reference walking speed.

**Design decision**
- `0.30 m/s`: low-speed validation
- `0.40 m/s`: intermediate speed
- `0.45 m/s`: **current nominal walking speed**
- `0.50 m/s`: upper/reference gait

---

### 2026-10-02 — Collision-based foot-clearance measurement

**Problem**
- Measuring ankle-body vertical movement did not represent actual foot clearance from the floor.

**MJCF analysis**
- Located the left/right `ankle_roll` collision boxes.
- Foot collision geometry is represented by an oriented MuJoCo box.

**Changes**
- Added automatic foot collision-geometry detection.
- Calculated the lowest point of each oriented collision box from:
  - geom world position
  - geom world rotation matrix
  - collision box half-extents
- Measured the lowest collision point relative to the ground plane.
- Added body-heading-frame fore-aft foot excursion measurement.

**Result**
- Gait analysis now uses actual collision geometry rather than ankle-link height.
- Representative `0.45 m/s` tests produced foot clearance in approximately the low-to-mid `3 cm` range.
- The metric can later be compared against physical foot clearance during Sim2Real testing.

---

### 2026-10-02 — Heading-controller tuning

**Initial controller**
- heading `Kp = 0.4`
- heading `Kd = 0.05`
- path-heading gain = `1.2`
- path-heading limit = `12 deg`

**Experiment 1: weaker path correction**
- path-heading gain reduced:
  - `1.2 -> 0.6`
- heading limit reduced:
  - `12 deg -> 6 deg`

**Result**
- Cross-track error continuously accumulated.
- Drift reached approximately `0.85 m`.
- The `6 deg` heading limit was insufficient to recover from lateral drift.

**Engineering decision**
- Reverted path-heading control to:
  - gain = `1.2`
  - limit = `12 deg`

**Experiment 2: stronger heading feedback**
- Increased heading proportional gain:
  - `Kp: 0.4 -> 0.7`
- Kept:
  - `Kd = 0.05`

**Result**
- Cross-track error no longer diverged.
- Representative `0.45 m/s` testing maintained cross-track error around approximately `0.09–0.15 m`.
- Mean forward motion over the evaluation window was close to the `0.45 m/s` command, although instantaneous velocity variation remained.

---

### 2026-10-02 — Direct lateral path correction

**Problem**
- Heading control prevented divergence, but the robot still maintained a noticeable lateral offset from the reference path.

**Approach**
- Added a small proportional lateral velocity command in addition to heading correction.

**Controller**
- lateral gain = `0.20`
- lateral command limit = `±0.03 m/s`

Positive cross-track error generates a small negative `vy` command to move the humanoid toward the reference line.

Manual `A/D` or `Q/E` commands override automatic path control, and a new reference line is created after manual input.

**Representative test result**
- `policy vy` remained small, generally around `-0.01` to `-0.02 m/s`.
- The lateral command did not reach the `±0.03 m/s` saturation limit.
- Cross-track error was reduced to roughly `0.05–0.11 m`.
- Foot clearance remained in the low-to-mid `3 cm` range.
- The additional path controller therefore improved tracking without destroying the learned stepping gait.

**Engineering decision**
- Keep the lateral controller deliberately weak so that path correction does not dominate the learned RL locomotion policy.

---

### 2026-10-02 — Physics-rate actuator validation

**Goal**
- Verify whether the previously observed torque saturation was a 25 Hz diagnostic sampling artifact or an actual actuator-limit behavior inside the 2000 Hz MuJoCo physics loop.

**Method**
- Temporarily instrumented the PD controller before and after torque clipping.
- Recorded raw PD torque demand, clipped torque, peak torque, and saturation duty at the 2000 Hz physics-substep rate.
- The configured leg effort limit remained `6.0 Nm`.

**Result**
- The previously observed saturation was confirmed at physics-substep resolution.
- Ankle-pitch joints repeatedly showed approximately `35–40%` saturation.
- Hip-yaw joints repeatedly showed approximately `30%` saturation.
- Raw PD demand occasionally exceeded `20 Nm` in ankle-pitch and knee joints before the `6 Nm` clipping stage.
- Ankle-roll joints showed essentially no saturation.

**Interpretation**
- The torque clipping is therefore not simply an artifact of the earlier 25 Hz diagnostic.
- The main recurrent actuator-demand bottlenecks are ankle pitch and hip yaw under the current MuJoCo PD/effort-limit configuration.
- This does not by itself prove that the hardware torque limit should be increased; physical actuator limits and the training-side actuator model must be checked before changing the limit.

**Engineering decision**
- Retain the result as a Sim2Real risk item.
- Remove the physics-rate diagnostic from the normal simulator because per-substep logging caused substantial runtime overhead.
- Return to lightweight policy-rate diagnostics and continue with contact-based gait analysis.

---

## Current Baseline — 2026-10-02

### Forward velocity commands

| Stage | Velocity | Purpose |
|---|---:|---|
| Low-speed validation | `0.30 m/s` | Minimum walking evaluation |
| Intermediate | `0.40 m/s` | Transition gait |
| Nominal | **`0.45 m/s`** | Current Sim2Real target |
| Upper/reference | `0.50 m/s` | Cleanest simulated gait |

### Walk-start controller

- threshold: `0.30 m/s`
- temporary startup policy command: `0.40 m/s`
- duration: `1.0 s`

### Heading controller

- `Kp = 0.7`
- `Kd = 0.05`
- yaw correction limit = `0.18 rad/s`

### Path controller

- path-heading gain = `1.2`
- path-heading limit = `12 deg`
- lateral gain = `0.20`
- lateral velocity limit = `±0.03 m/s`

### Runtime diagnostics

- requested `vx`
- policy `vx`
- policy `vy`
- cross-track error
- path-heading correction
- world `vx / vy`
- body-frame forward velocity
- yaw
- yaw error
- yaw command
- yaw rate
- passive-arm torque RMS
- passive-arm saturation
- base angular speed
- left/right collision-based foot clearance
- left/right fore-aft foot excursion

---

## Validation

Development environment:
- Python 3.11
- PyTorch 2.7.0 + CUDA 12.8
- Isaac Sim 5.1.0
- Isaac Lab 2.3.2
- MuJoCo 3.3.5
- ONNX Runtime 1.22.1

Validation completed:
- Python compilation checks passed.
- `git diff --check` passed.
- Full humanoid model launched successfully.
- Keyboard locomotion controls confirmed.
- WSLg D3D12 rendering confirmed.
- `0.30 / 0.40 / 0.45 / 0.50 m/s` command ladder confirmed.
- Walk-start boost confirmed.
- Heading/path controller tested.
- Collision-based foot-clearance diagnostics confirmed.
- Direct lateral correction tested at the `0.45 m/s` nominal gait.

---

## Latest Checkpoint

### 2026-10-02

`571029e Tune walking gait and path tracking for sim2real`

This checkpoint includes:
- discrete walking-speed control,
- low-speed startup boost,
- heading/path tuning,
- small direct lateral correction,
- gait/path diagnostics,
- collision-based foot-clearance measurement.

---

## Remaining Sim2Real Work

Before deployment to the physical humanoid:

- characterize instantaneous forward-speed variation at `0.45 m/s`,
- investigate the remaining left/right gait asymmetry,
- validate heading/lateral-control gains under real IMU noise and actuator limits,
- verify startup, stopping, and recovery conservatively on hardware,
- compare simulated and physical foot-clearance behavior,
- validate motor torque/current margins,
- evaluate floor-friction sensitivity,
- test small obstacles/thresholds only after flat-ground hardware validation,
- determine whether additional policy retraining is required after Sim2Real testing.

The current PR represents a stable **Sim2Sim locomotion baseline and Sim2Real preparation checkpoint**, not final hardware tuning.

---

### 2026-10-02 — Contact-based gait validation

**Goal**
- Determine whether the previously observed left/right fore-aft difference represented a real gait asymmetry.

**Method**
- Added lightweight MuJoCo foot-ground contact detection at the 25 Hz policy rate.
- Detected touchdown and liftoff events.
- Measured step time, step length, stride time, stride length, stance time, swing time, and left/right asymmetry.

**Result**
- At the nominal `0.45 m/s` command:
  - mean actual forward velocity: approximately `0.40 m/s`
  - step time: approximately `0.26 s`
  - step length: approximately `10.2 cm`
  - stride time: approximately `0.51 s`
  - stride length: approximately `20.3 cm`
  - stance time: approximately `0.36 s`
  - swing time: approximately `0.15 s`
- Mean stride-length asymmetry was approximately `1.8%`.
- Mean stride-time asymmetry was approximately `0.9%`.
- Step-derived forward velocity was consistent with the independently measured base forward velocity.

**Interpretation**
- The locomotion gait is substantially left/right symmetric at the nominal walking speed.
- The previously observed difference in the body-relative foot fore-aft excursion metric did not correspond to a comparable touchdown-to-touchdown stride asymmetry.

**Engineering decision**
- Accept the current `0.45 m/s` gait as the nominal Sim2Sim baseline.
- Keep lightweight contact-based gait diagnostics.
- Proceed to Sim2Real robustness testing rather than further nominal-gait tuning.

---

### 2026-10-02 — Floor-friction robustness validation

**Goal**
- Evaluate sensitivity of the nominal walking policy to foot-ground friction changes before hardware deployment.

**Method**
- Added runtime MuJoCo friction scaling without modifying the source MJCF.
- Evaluated three conditions at the nominal `0.45 m/s` command:
  - low: `0.5x`
  - nominal: `1.0x`
  - high: `1.5x`
- Compared forward velocity, path tracking, orientation error, gait timing, stride length, foot clearance, and left/right gait symmetry.

**Result**
- Nominal (`1.0x`):
  - forward velocity: approximately `0.400 m/s`
  - cross-track RMS: approximately `0.103 m`
  - yaw-error RMS: approximately `4.38 deg`
  - stride length: approximately `20.1 cm`
  - stride-length asymmetry: approximately `2.0%`

- Low friction (`0.5x`):
  - forward velocity decreased to approximately `0.382 m/s`
  - cross-track RMS increased to approximately `0.154 m`
  - yaw-error RMS increased to approximately `6.99 deg`
  - stride length decreased to approximately `17.0 cm`
  - mean stride-length asymmetry increased to approximately `12.4%`
  - transient stride asymmetry exceeded `50%`
  - locomotion continued without falling, but nominal tracking and gait symmetry were not maintained.

- High friction (`1.5x`):
  - forward velocity increased to approximately `0.434 m/s`
  - cross-track RMS decreased to approximately `0.082 m`
  - yaw-error RMS decreased to approximately `2.70 deg`
  - stride length increased to approximately `24.8 cm`
  - stride-length asymmetry decreased to approximately `0.3%`

**Interpretation**
- The policy remains stable over substantial friction variation, but low-friction conditions produce significant path-tracking and gait-symmetry degradation.
- Higher friction shifts the locomotion system toward a longer-stride, slower-cadence gait while maintaining stable left/right symmetry.
- Aggregate leg torque saturation remained approximately `16–17%` across all three conditions, indicating that the low-friction degradation was not accompanied by increased actuator clipping.

**Engineering decision**
- Accept the friction test as sufficient for the current Sim2Real preparation stage.
- Treat low-friction surfaces as a hardware validation risk.
- Proceed to actuator-delay robustness testing.

---

### 2026-10-02 — Action-delay robustness validation

**Goal**
- Evaluate locomotion sensitivity to control latency before hardware deployment.

**Method**
- Added runtime policy-action delay using a discrete FIFO queue.
- Policy frequency was 25 Hz, corresponding to a 40 ms action period.
- Tested:
  - `0 ms` / 0 policy steps
  - `40 ms` / 1 policy step
  - `80 ms` / 2 policy steps
- Floor friction remained at the nominal `1.0x` condition.

**Result**
- `0 ms`:
  - sustained stable locomotion
  - mean forward velocity approximately `0.402 m/s`
  - forward-velocity std approximately `0.206 m/s`
  - path-lateral RMS approximately `0.136 m/s`
  - roll/pitch RMS approximately `1.20 / 1.78 deg`
  - leg-torque saturation approximately `16.6%`

- `40 ms`:
  - locomotion initially continued, but sustained stability was lost
  - pre-failure mean forward velocity approximately `0.435 m/s`
  - forward-velocity std increased to approximately `0.814 m/s`
  - path-lateral RMS increased to approximately `0.450 m/s`
  - roll/pitch RMS increased to approximately `2.74 / 3.62 deg`
  - leg-torque saturation increased to approximately `22.7%`
  - stride length increased to approximately `27 cm`
  - gait asymmetry increased
  - catastrophic orientation loss occurred after roughly 10–15 s of walking.

- `80 ms`:
  - stable walking gait was not established
  - large orientation errors appeared almost immediately
  - torque saturation increased to approximately `50%`
  - the robot rapidly lost balance.

**Interpretation**
- The current locomotion policy is substantially more sensitive to action latency than to the tested friction variation.
- A one-policy-step (`40 ms`) delay is already outside the sustained-stability region for the current controller.
- The exact latency stability boundary cannot be determined from this discrete 25 Hz delay test because the delay is quantized in 40 ms increments.

**Engineering decision**
- Treat end-to-end control latency as a major Sim2Real risk.
- Keep the runtime delay test for future controller validation.
- Measure actual inference, communication, motor-controller, and actuator latency on hardware.
- If necessary, perform finer sub-policy-period delay testing or retraining with latency randomization.
