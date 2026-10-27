"""Closed-loop, single-lane road demonstrations; no object recognition is claimed."""
import numpy as np
from simulation import KalmanFilter, time_to_collision


def run_road(c):
    t = np.arange(int(np.floor(c.duration / c.dt)) + 1) * c.dt
    n = len(t)
    rng = np.random.default_rng(c.seed)
    # Same noise sequence is used for repeatable experiments.
    noise = rng.normal(0, c.sensor_std, n)
    ego, lead = np.zeros(n), np.zeros(n)
    ego_speed, lead_speed = np.zeros(n), np.zeros(n)
    ego_speed[0] = 18.
    lead[0] = 50. if c.scenario == 'following' else 100.
    lead_speed[:] = 18. - 3. * np.clip(t - 4., 0., 4.) if c.scenario == 'following' else 0.
    lead[1:] = lead[0] + np.cumsum(.5 * (lead_speed[:-1] + lead_speed[1:]) * c.dt)
    measurement = np.empty(n)
    estimate, covariance = np.empty((n, 2)), np.empty((n, 2, 2))
    innovation, acceleration = np.empty(n), np.zeros(n)
    kf = KalmanFilter(c.dt, c.sensor_std, c.accel_std)
    roadside = np.array([[30., 5.], [75., -5.], [125., 5.], [180., -5.]])
    roadside_readings = np.full((n, len(roadside)), np.nan)
    for k in range(n):
        gap = lead[k] - ego[k]
        measurement[k] = gap + noise[k] if 0 <= gap <= 180 else np.nan
        # Known ego acceleration is a control input, not ground-truth lead motion.
        actual_ego_accel = (ego_speed[k] - ego_speed[k-1])/c.dt if k else 0.
        estimate[k], covariance[k], innovation[k] = kf.step(measurement[k], -actual_ego_accel)
        # Roadside objects are detected too, but lane metadata excludes them from control.
        forward = roadside[:, 0] - ego[k]
        ranges = np.hypot(forward, roadside[:, 1])
        visible = (forward >= 0) & (ranges <= 180)
        roadside_readings[k, visible] = ranges[visible] + rng.normal(0, c.sensor_std, visible.sum())
        d, v = estimate[k]
        ttc = float(time_to_collision(d, v))
        # Conservative demonstration controller: brake while closing at short TTC.
        # A known stationary target keeps braking until stopped. No truth gap is used.
        braking = (ttc < 5. and v < -.3) or (c.scenario == 'obstacle' and d < 12.)
        acceleration[k] = -4. if braking and ego_speed[k] > 0 else 0.
        if k + 1 < n:
            ego_speed[k+1] = max(0., ego_speed[k] + acceleration[k]*c.dt)
            ego[k+1] = ego[k] + .5*(ego_speed[k] + ego_speed[k+1])*c.dt
    truth = np.column_stack((lead - ego, lead_speed - ego_speed))
    return dict(config=c, t=t, ego=ego, lead=lead, ego_speed=ego_speed,
                lead_speed=lead_speed, acceleration=acceleration, truth=truth,
                measurement=measurement, estimate=estimate, covariance=covariance,
                innovation=innovation, raw_speed=np.r_[np.nan, np.diff(measurement)/c.dt],
                ttc=time_to_collision(estimate[:, 0], estimate[:, 1]),
                true_ttc=time_to_collision(truth[:, 0], truth[:, 1]),
                roadside=roadside, roadside_readings=roadside_readings)


def animate_road(r, directory, gif=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    fig, (ax, chart) = plt.subplots(2, 1, figsize=(13, 7), height_ratios=[1, 1.2], constrained_layout=True)
    end = max(r['lead'].max(), r['ego'].max(), 180) + 20
    ax.set(xlim=(-10, end), ylim=(-8, 13), xlabel='Fixed world position along road (m)', yticks=[])
    ax.axhspan(-2, 2, color='#e2e4e8')
    for x in np.arange(0, end, 12):
        ax.plot([x, x+5], [0, 0], color='white', linewidth=2)
    for x, y in r['roadside']:
        ax.add_patch(Rectangle((x-1, y-.8), 2, 1.6, color='gray'))
    detected, = ax.plot([], [], 'mo', fillstyle='none', markersize=12, label='Detected roadside objects (outside lane)')
    ego = Rectangle((-4.5, -.8), 4.5, 1.6, color='tab:blue')
    target = Rectangle((r['lead'][0], -.8), 4.5, 1.6, color='tab:green')
    ax.add_patch(ego)
    ax.add_patch(target)
    beam, = ax.plot([], [], color='cyan', linewidth=2, label='In-lane range measurement')
    measured, = ax.plot([], [], 'rx', label='Noisy measured target position')
    estimated, = ax.plot([], [], '|', color='darkorange', markersize=22, label='KF target position')
    info = ax.text(.01, .97, '', transform=ax.transAxes, va='top', fontsize=9, family='monospace')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title(f"{r['config'].scenario}: blue ego moves in world coordinates; {'green lead brakes' if r['config'].scenario == 'following' else 'green obstacle stays fixed'}")
    chart.plot(r['t'], r['truth'][:, 0], 'k--', label='True gap')
    chart.plot(r['t'], r['measurement'], '.', alpha=.35, label='Noisy range sensor')
    chart.plot(r['t'], r['estimate'][:, 0], label='KF gap')
    cursor = chart.axvline(0, color='red', alpha=.6)
    from simulation import metrics
    scores = metrics(r)
    chart.set(xlabel='Time (s)', ylabel='Gap (m)', title=f"Same sensor data: distance RMSE raw {scores['distance_sensor_rmse_m']:.2f} m -> KF {scores['distance_kf_rmse_m']:.2f} m")
    chart.grid(alpha=.2)
    chart.legend(loc='upper right')
    # Solve the static layout once rather than recalculating it at every frame.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    stride = max(1, int(np.ceil(.1/r['config'].dt)))
    def update(k):
        x = r['ego'][k]
        ego.set_x(x-4.5)
        target.set_x(r['lead'][k])
        measured.set_data([x+r['measurement'][k]], [0])
        estimated.set_data([x+r['estimate'][k, 0]], [0])
        beam.set_data([x, x+r['measurement'][k]], [0, 0])
        visible = np.isfinite(r['roadside_readings'][k])
        detected.set_data(r['roadside'][visible, 0], r['roadside'][visible, 1])
        braking = r['acceleration'][k] < 0
        ego.set_color('tab:red' if braking else 'tab:blue')
        ttc = r['ttc'][k]
        text = f'{ttc:.1f}s' if np.isfinite(ttc) else 'not closing'
        info.set_text(f"t={r['t'][k]:4.1f}s | ego={r['ego_speed'][k]:4.1f}m/s | KF gap={r['estimate'][k,0]:5.1f}m\nrelative speed={r['estimate'][k,1]:5.1f}m/s | TTC={text} | {'BRAKING' if braking else 'COASTING / STOPPED'}")
        cursor.set_xdata([r['t'][k]]*2)
        return ego, target, measured, estimated, beam, detected, info, cursor
    anim = FuncAnimation(fig, update, frames=range(0, len(r['t']), stride), interval=1000*r['config'].dt*stride, repeat=True, blit=False)
    if gif:
        anim.save(directory/'simulation.gif', writer=PillowWriter(fps=1/(r['config'].dt*stride)), dpi=70)
    return fig, anim


def plot_road_comparison(r, directory):
    import matplotlib.pyplot as plt
    from simulation import metrics
    m = metrics(r)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    t = r['t']
    axes[0].plot(t, r['truth'][:, 0], 'k--', label='True gap')
    axes[0].plot(t, r['measurement'], '.', alpha=.4, label='Noisy sensor')
    axes[0].plot(t, r['estimate'][:, 0], label='KF estimate')
    axes[0].set_ylabel('Distance (m)')
    axes[0].set_title(f"Distance RMSE: sensor {m['distance_sensor_rmse_m']:.2f} m; KF {m['distance_kf_rmse_m']:.2f} m")
    axes[1].plot(t, r['raw_speed'], '.', alpha=.25, label='Raw distance differences')
    axes[1].plot(t, r['truth'][:, 1], 'k--', label='True relative speed')
    axes[1].plot(t, r['estimate'][:, 1], label='KF relative speed')
    axes[1].set_ylabel('Relative speed (m/s)')
    axes[1].set_title(f"Speed RMSE: differences {m['relative_speed_difference_rmse_mps']:.2f} m/s; KF {m['relative_speed_kf_rmse_mps']:.2f} m/s")
    axes[2].plot(t, r['ego_speed'], label='Ego speed')
    axes[2].plot(t, r['lead_speed'], label='Target speed')
    axes[2].fill_between(t, 0, max(r['ego_speed'].max(), 1), where=r['acceleration'] < 0, alpha=.15, color='red', label='Estimated-state braking command')
    axes[2].set_ylabel('Vehicle speed (m/s)')
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f"{r['config'].scenario}: KF estimation benefit and controller response")
    fig.savefig(directory/'kf_comparison.png', dpi=140)
    return fig
