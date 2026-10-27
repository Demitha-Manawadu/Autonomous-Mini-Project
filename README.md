# Distance and Relative Speed Estimation for Autonomous Vehicle Safety

A Python mini-project implementing a linear Kalman filter from first principles. Two road demonstrations show both vehicles moving in fixed world coordinates, or your vehicle approaching a stationary obstacle. A simulated range sensor supplies noisy readings; KF estimates drive a simple braking controller. The original five estimation experiments remain available.

## Run on Windows

Open PowerShell in this folder, then run:

```powershell
python -m pip install -r requirements.txt
python main.py --demo
```

Four windows open: a KF comparison graph and road animation for each scenario. Close all windows to exit. After installing dependencies, you can also double-click **Run Simulation.bat**. `python main.py` alone opens the following scenario. If Windows uses `py` instead of `python`, substitute `py` in the commands. Python 3.10 or newer is recommended.

## The two road demonstrations

```powershell
python main.py --scenario following
python main.py --scenario obstacle
python main.py --demo --headless --gif
```

**Following:** ego and lead start at 18 m/s with a 50 m gap. The lead brakes from 4–8 seconds to 6 m/s. Ego uses estimated distance and closing speed to brake. Both rectangles move along a fixed road; the lead stays ahead. This is a simple brake-only demonstration: ego does not accelerate again to maintain a desired following gap.

**Obstacle:** ego starts at 18 m/s and approaches a stationary in-lane obstacle at world position 100 m. Estimated collision risk triggers braking. The obstacle stays in one place while ego moves and stops. The controller is deliberately simple and is not a validated emergency-braking design.

Four gray roadside obstacles are also simulated. Magenta rings highlight visible objects; noisy range readings are saved in `road_sensor.csv`. The sensor has a simplified forward 180 m range. Object identities and lane membership are supplied by the simulator: this is not camera recognition, a detailed radar model, or automatic lane detection. Roadside objects are outside the lane and do not enter the in-lane object's KF. There is one tracked in-lane target, so no multi-object association algorithm is needed.

In the road window, blue is ego and it turns red when braking. Cyan shows the in-lane sensor beam, red crosses show noisy measured target position, and the orange marker shows estimated target position. The lower panel compares true, measured, and filtered gap. The separate comparison window shows distance, relative speed, and vehicle speeds; red shading marks braking. `kf_comparison.png` saves this view. `performance.png` retains the detailed uncertainty/error analysis.

### How this demonstrates the importance of KF

Compare the noisy readings and KF line against the dashed true trajectory. The displayed RMSE measures actual error, so smoothing alone is not treated as proof. Compare relative speed from raw distance differences with KF speed: direct differences amplify sensor noise. These are estimates from the **same measurements on the same driven trajectory**, not separate raw-controlled and KF-controlled vehicles. Better estimation does not by itself prove better braking safety.

With default settings, the 20-seed road experiments gave mean distance RMSE of **2.007 m raw versus 1.206 m KF** for following and **2.007 m raw versus 0.697 m KF** for the fixed obstacle. Mean speed RMSE was about **28.7 m/s from differences versus 1.73 and 2.04 m/s with KF**, respectively. Initial speed uncertainty is included. Reproduce with `python main.py --demo --headless --trials 20 --output outputs/road_trials`.

### Is running a KF difficult?

This particular KF is computationally small: it stores two state values and a 2×2 covariance matrix. Every 0.1 seconds it predicts movement, then corrects the prediction using one range reading. No neural-network training or GPU is needed. A local timing check of 10,000 updates averaged about 17 microseconds per update, excluding plotting; actual speed depends on the computer. Graphics and GIF export take much more time than this filter. The difficult engineering work is choosing a suitable model, tuning noise, handling sensor failures, and validating the result. Here we simplify that work by using known sensor noise, known object identities, one lane, and synthetic motion.

The simulation generates motion using `position += average_speed * dt`, then creates readings using `measured_gap = true_gap + random_noise`. Truth is used to generate the sensor and score estimates. The KF receives only that measurement and, in the road demonstrations, known ego acceleration as a control input. It never receives the true target speed or gap. Positive ego acceleration reduces relative acceleration, so the prediction adds `G * (-ego_acceleration)`; unknown lead acceleration remains process uncertainty. Ego acceleration comes from the simulated executed speed change, including stopping saturation.

The demonstration controller commands −4 m/s² when estimated TTC is below 5 s and estimated closing speed exceeds 0.3 m/s. For the known stationary obstacle, estimated gap below 12 m also maintains braking. Speed is limited to zero. These are illustrative choices; changing noise, time step, or motion can change stopping performance.

```powershell
# Select a scenario
python main.py --scenario dropout

# Generate all seven scenario plots and data without opening windows
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
| following | Closed-loop moving-road demonstration with braking lead vehicle |
| obstacle | Closed-loop moving-road demonstration with stationary in-lane obstacle |
| cruise | Both vehicles travel at 20 m/s, initial gap 65 m |
| approach | Ego 20 m/s, lead 18 m/s, initial gap 65 m |
| braking | Initial gap 35 m; lead decelerates from 20 to 14 m/s at 6–8 s; ego does the same at 10–12 s |
| noisy | Same braking trajectory, with sensor noise standard deviation multiplied by three |
| dropout | Same braking trajectory, with no measurements from 8 s up to, but excluding, 10 s |

The original five experiments use scripted ego trajectories; only `following` and `obstacle` use estimated-state feedback braking. Default runs last 25 s at 10 Hz. Extending approach beyond 32.5 s creates a closed gap; this simple model does not simulate collision physics.

Each scenario directory contains `performance.png`, `results.csv`, and `metrics.json`; `--gif` adds `simulation.gif`. Repeated trials use seed subfolders and plot/export only the first seed. `summary.csv` has one row per scenario and seed. CSV missing readings are `nan`; infinite TTC means the estimated vehicles are not closing faster than 0.1 m/s. The TTC plot omits infinity and displays only 0–30 seconds.

The original five experiments retain the ego-relative animation. Blue is ego, green/orange/red is the lead vehicle, orange marker is estimated lead position, and red crosses are sensor readings. Caution is TTC <5 s; warning is TTC <3 s. These thresholds are illustrative, not validated automotive settings. The two new road demonstrations use the fixed world view described above.

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
- `road.py`: road scenarios, sensor objects, feedback braking, world-coordinate animation.
- `test_simulation.py`: mathematical and scenario checks.
- `REPORT.md`: methodology and reproducible experimental findings.

For your submission, explain the application and assumptions, derive the model, describe predict/update and implementation, present scenario plots and RMSE results, discuss tuning and failure cases, and conclude with limitations. This project uses synthetic data and demonstrates estimation; it is not an autonomous driving system.
