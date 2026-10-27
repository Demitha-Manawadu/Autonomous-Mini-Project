# Distance and Relative Speed Estimation for Autonomous Vehicle Safety

A Python mini-project implementing a linear Kalman filter from first principles. A simulated front-facing sensor measures the bumper-to-bumper gap to a lead vehicle. The filter estimates gap and relative speed; an animation illustrates a time-to-collision (TTC) warning.

## Run on Windows

Open PowerShell in this folder, then run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Two windows open: performance graphs and a repeating vehicle animation. Close both to exit. After installing dependencies, you can also double-click **Run Simulation.bat**. If Windows uses `py` instead of `python`, substitute `py` in the commands. Python 3.10 or newer is recommended.

```powershell
# Select a scenario
python main.py --scenario dropout

# Generate all scenario plots and data without opening windows
python main.py --all --headless

# Export an animation viewable without Python (Pillow required; takes time)
python main.py --scenario braking --headless --gif

# Repeated experiments with 20 independent noise seeds per scenario
python main.py --all --headless --trials 20 --output outputs/repeated

# Change sensor and motion uncertainty
python main.py --scenario braking --sensor-std 3 --accel-std 4

# Verification
python -m unittest -v
```

Use `python main.py --help` for all options. Commands save results relative to the current folder. Reusing an output folder replaces matching files; use `--output` to preserve experiments.

## Scenarios and outputs

| Scenario | Motion / measurement behaviour |
|---|---|
| cruise | Both vehicles travel at 20 m/s, initial gap 65 m |
| approach | Ego 20 m/s, lead 18 m/s, initial gap 65 m |
| braking | Initial gap 35 m; lead decelerates from 20 to 14 m/s at 6–8 s; ego does the same at 10–12 s |
| noisy | Same braking trajectory, with sensor noise standard deviation multiplied by three |
| dropout | Same braking trajectory, with no measurements from 8 s up to, but excluding, 10 s |

These are scripted vehicle trajectories, not a feedback braking controller. Ego response is scheduled in advance. Default runs last 25 s at 10 Hz and do not collide. Extending approach beyond 32.5 s creates a closed gap; this simple model does not simulate collision physics.

Each scenario directory contains `performance.png`, `results.csv`, and `metrics.json`; `--gif` adds `simulation.gif`. Repeated trials use seed subfolders and plot/export only the first seed. `summary.csv` has one row per scenario and seed. CSV missing readings are `nan`; infinite TTC means the estimated vehicles are not closing faster than 0.1 m/s. The TTC plot omits infinity and displays only 0–30 seconds.

The road animation uses ego-relative coordinates. Blue is ego, green/orange/red is the lead vehicle, orange marker is estimated lead position, and red crosses are sensor readings. Caution is TTC <5 s; warning is TTC <3 s. These thresholds are illustrative, not validated automotive settings.

## Mathematical implementation

State `x = [d, v_rel]ᵀ`, where `v_rel = v_lead - v_ego`; negative means closing. Distance is measured from ego front bumper to lead rear bumper. Truth trajectories use trapezoidal integration of scripted speeds. Sensor samples are `z = d + n`, with independent zero-mean Gaussian noise.

The linear model is:

```text
F = [[1, dt], [0, 1]]       H = [[1, 0]]
G = [dt²/2, dt]ᵀ           Q = sigma_a² G Gᵀ
R = sigma_z²
```

`sigma_a` models an independent random relative acceleration held constant over each sample interval. It is a process tuning parameter, not a measured vehicle acceleration. This discrete acceleration model differs from a continuous white acceleration spectral-density model; retune it if changing the sampling interval.

Prediction: `x⁻ = F x`, `P⁻ = F P Fᵀ + Q`.

Correction: `y = z - H x⁻`, `S = H P⁻ Hᵀ + R`, `K = P⁻ Hᵀ / S`, `x = x⁻ + K y`.

Covariance uses the Joseph form: `P = (I-KH) P⁻ (I-KH)ᵀ + K R Kᵀ`. At missing samples, only prediction runs. Initialization uses the first available distance, zero relative speed, and `P = diag(R, 100)`; the first measurement is not counted twice. Ground truth never enters the filter. Measurement noise is known from the simulated sensor configuration, including the noisier scenario.

`H` and `HF` stacked have rank two for positive `dt`: relative speed is observable from distance changes over time. A standard KF therefore suffices for this longitudinal linear model. Turning and nonlinear range-bearing tracking would require a different model.

For positive distance and relative speed below −0.1 m/s, `TTC = d / (-v_rel)`. Non-closing states return infinity; a closed gap returns zero. TTC assumes unchanged relative speed and is less reliable during acceleration. It is calculated after filtering, not used as a linear KF measurement.

## Evaluation and interpretation

RMSE is `sqrt(mean((estimate - truth)²))`. Distance is compared with raw measurements; speed is compared with the backward difference `(z[k]-z[k-1])/dt`. The latter strongly amplifies noise and is a simple baseline. Metrics include full-run KF errors and matched-sample KF errors for fair comparisons when sensor readings are absent. No unavailable sensor measurements are filled with truth.

Shaded bands are nominal Gaussian 95% intervals, not guaranteed coverage. The scripted accelerations deliberately violate the constant-velocity expectation. Study transient bias during braking, recovery after dropout, and empirical coverage. A KF need not improve every metric in every scenario: prediction during an outage can drift, especially when braking has not yet been identified.

Increase `--accel-std` to permit faster changes, usually at the cost of noisier estimates. Increase assumed measurement noise to trust measurements less. Here `--sensor-std` changes both the generated sensor noise and its correctly specified filter R; it does not independently test incorrect R. For a tuning study, hold sensor noise fixed and vary `--accel-std`.

## Files and report structure

- `simulation.py`: motion, sensor, filter, TTC, and evaluation.
- `main.py`: command line, plots, animation, CSV/JSON output.
- `test_simulation.py`: mathematical and scenario checks.
- `REPORT.md`: methodology and reproducible experimental findings.

For your submission, explain the application and assumptions, derive the model, describe predict/update and implementation, present scenario plots and RMSE results, discuss tuning and failure cases, and conclude with limitations. This project uses synthetic data and demonstrates estimation; it is not an autonomous driving system.
