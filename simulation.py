"""Reproducible longitudinal vehicle tracking with a linear Kalman filter."""
from dataclasses import dataclass
import numpy as np

SCENARIOS = ('following', 'obstacle', 'cruise', 'approach', 'braking', 'noisy', 'dropout')


@dataclass
class Config:
    scenario: str = 'braking'
    duration: float = 25.0
    dt: float = 0.1
    sensor_std: float = 2.0
    accel_std: float = 2.0
    seed: int = 42

    def validate(self):
        if self.scenario not in SCENARIOS:
            raise ValueError('Unknown scenario')
        values = (self.duration, self.dt, self.sensor_std, self.accel_std)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Duration, step, and noise standard deviations must be finite and positive')
        if self.duration < self.dt:
            raise ValueError('Duration must be at least one time step')


class KalmanFilter:
    def __init__(self, dt, sensor_std, accel_std):
        self.F = np.array([[1., dt], [0., 1.]])
        self.H = np.array([[1., 0.]])
        # Independent, constant relative acceleration within each sample interval.
        g = np.array([0.5 * dt**2, dt])
        self.G = g
        self.Q = accel_std**2 * np.outer(g, g)
        self.R = sensor_std**2
        self.x = None
        self.P = np.diag([self.R, 100.])

    def step(self, measurement, relative_acceleration=0.):
        if self.x is None:
            if not np.isfinite(measurement):
                raise ValueError('First measurement must be available')
            self.x = np.array([measurement, 0.])
            return self.x.copy(), self.P.copy(), np.nan
        self.x = self.F @ self.x + self.G * relative_acceleration
        self.P = self.F @ self.P @ self.F.T + self.Q
        innovation = np.nan
        if np.isfinite(measurement):
            innovation = measurement - self.x[0]
            s = self.P[0, 0] + self.R
            gain = self.P[:, 0] / s
            self.x += gain * innovation
            # Joseph form preserves covariance symmetry and positive semidefiniteness.
            a = np.eye(2) - np.outer(gain, self.H[0])
            self.P = a @ self.P @ a.T + self.R * np.outer(gain, gain)
        return self.x.copy(), self.P.copy(), innovation


def time_to_collision(distance, relative_speed):
    d, v = np.broadcast_arrays(np.asarray(distance, float), np.asarray(relative_speed, float))
    result = np.full(d.shape, np.inf)
    np.divide(d, -v, out=result, where=(d > 0) & (v < -0.1))
    return np.where(d <= 0, 0., result)


def run(config):
    config.validate()
    if config.scenario in ('following', 'obstacle'):
        from road import run_road
        return run_road(config)
    c = config
    t = np.arange(int(np.floor(c.duration / c.dt)) + 1) * c.dt
    n = len(t)
    ego_speed = np.full(n, 20.)
    lead_speed = np.full(n, 20.)
    initial_gap = 65.
    if c.scenario == 'approach':
        lead_speed[:] = 18.
    if c.scenario in ('braking', 'dropout', 'noisy'):
        initial_gap = 35.
        # Lead brakes at 6 s; ego reacts later. No controller uses ground truth.
        lead_speed = 20. - 3. * np.clip(t - 6., 0., 2.)
        ego_speed = 20. - 3. * np.clip(t - 10., 0., 2.)
    ego = np.zeros(n)
    lead = np.full(n, initial_gap)
    ego[1:] = np.cumsum((ego_speed[:-1] + ego_speed[1:]) * 0.5 * c.dt)
    lead[1:] = initial_gap + np.cumsum((lead_speed[:-1] + lead_speed[1:]) * 0.5 * c.dt)
    truth = np.column_stack((lead - ego, lead_speed - ego_speed))
    noise_std = c.sensor_std * (3. if c.scenario == 'noisy' else 1.)
    measurement = truth[:, 0] + np.random.default_rng(c.seed).normal(0, noise_std, n)
    if c.scenario == 'dropout':
        measurement[(t >= 8.) & (t < 10.)] = np.nan
    kf = KalmanFilter(c.dt, noise_std, c.accel_std)
    estimate = np.empty((n, 2))
    covariance = np.empty((n, 2, 2))
    innovation = np.empty(n)
    for k, z in enumerate(measurement):
        estimate[k], covariance[k], innovation[k] = kf.step(z)
    raw_speed = np.r_[np.nan, np.diff(measurement) / c.dt]
    return dict(config=c, t=t, ego=ego, lead=lead, truth=truth,
                measurement=measurement, estimate=estimate, covariance=covariance,
                innovation=innovation, raw_speed=raw_speed,
                ttc=time_to_collision(estimate[:, 0], estimate[:, 1]),
                true_ttc=time_to_collision(truth[:, 0], truth[:, 1]))


def metrics(result):
    r = result
    error = r['estimate'] - r['truth']
    available = np.isfinite(r['measurement'])
    speed_available = np.isfinite(r['raw_speed'])
    def rmse(a):
        return float(np.sqrt(np.mean(a**2)))
    return {
        'distance_sensor_rmse_m': rmse((r['measurement'] - r['truth'][:, 0])[available]),
        'distance_kf_rmse_m': rmse(error[:, 0]),
        'relative_speed_difference_rmse_mps': rmse((r['raw_speed'] - r['truth'][:, 1])[speed_available]),
        'relative_speed_kf_rmse_mps': rmse(error[:, 1]),
        'distance_kf_rmse_on_available_samples_m': rmse(error[available, 0]),
        'relative_speed_kf_rmse_on_difference_samples_mps': rmse(error[speed_available, 1]),
        'distance_95pct_coverage': float(np.mean(np.abs(error[:, 0]) <= 1.96 * np.sqrt(r['covariance'][:, 0, 0]))),
        'missing_measurements': int(np.sum(~available)),
    }
