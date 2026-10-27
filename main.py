"""CLI, saved results, plots, and vehicle animation. Run python main.py --help."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from simulation import Config, SCENARIOS, run, metrics


def save_results(r, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'metrics.json').open('w') as f:
        json.dump({'configuration': asdict(r['config']), 'metrics': metrics(r)}, f, indent=2)
    data = np.column_stack([r['t'], r['truth'], r['measurement'], r['estimate'],
                            r['raw_speed'], np.sqrt(r['covariance'][:, 0, 0]),
                            np.sqrt(r['covariance'][:, 1, 1]), r['ttc'], r['true_ttc']])
    np.savetxt(directory / 'results.csv', data, delimiter=',', comments='',
               header='time_s,true_distance_m,true_relative_speed_mps,measured_distance_m,estimated_distance_m,estimated_relative_speed_mps,difference_speed_mps,distance_std_m,speed_std_mps,estimated_ttc_s,true_ttc_s')
    if 'roadside_readings' in r:
        road_data = np.column_stack([r['t'], r['ego'], r['lead'], r['ego_speed'], r['lead_speed'], r['acceleration'], r['roadside_readings']])
        np.savetxt(directory / 'road_sensor.csv', road_data, delimiter=',', comments='', header='time_s,ego_front_m,target_rear_m,ego_speed_mps,target_speed_mps,ego_acceleration_mps2,roadside_1_range_m,roadside_2_range_m,roadside_3_range_m,roadside_4_range_m')


def plot_results(r, directory):
    import matplotlib.pyplot as plt
    t, truth, est = r['t'], r['truth'], r['estimate']
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for j, title in enumerate(('Distance (m)', 'Relative speed (m/s)')):
        ax = axes[0, j]
        ax.plot(t, r['measurement'] if j == 0 else r['raw_speed'], '.', alpha=.25, label='Sensor' if j == 0 else 'Distance differences')
        ax.plot(t, truth[:, j], 'k--', label='Ground truth')
        ax.plot(t, est[:, j], color='tab:blue', label='KF estimate')
        sigma = np.sqrt(r['covariance'][:, j, j])
        ax.fill_between(t, est[:, j] - 1.96*sigma, est[:, j] + 1.96*sigma, alpha=.2, label='Nominal 95% interval')
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(t, est[:, j] - truth[:, j], label='KF error')
        ax.fill_between(t, -1.96*sigma, 1.96*sigma, alpha=.2, label='±1.96 standard deviations')
        ax.axhline(0, color='k', linewidth=.6)
        ax.set_ylabel(title + ' error')
        ax.legend(fontsize=8)
    ax = axes[2, 0]
    for key, label in [('true_ttc', 'Ground truth'), ('ttc', 'KF estimate')]:
        ax.plot(t, np.where(np.isfinite(r[key]), r[key], np.nan), label=label)
    ax.axhline(5, color='orange', linestyle='--', label='Illustrative caution threshold')
    ax.set_ylim(0, 30)
    ax.set_ylabel('TTC (s), display capped at 30 s')
    ax.legend(fontsize=8)
    axes[2, 1].plot(t, np.sqrt(r['covariance'][:, 0, 0]), label='Distance uncertainty (m)')
    axes[2, 1].plot(t, np.sqrt(r['covariance'][:, 1, 1]), label='Speed uncertainty (m/s)')
    axes[2, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=.25)
        if r['config'].scenario == 'dropout':
            ax.axvspan(8, 10, color='gray', alpha=.15)
    fig.suptitle(f"Vehicle safety estimation — {r['config'].scenario}")
    fig.savefig(directory / 'performance.png', dpi=150)
    if 'roadside_readings' in r:
        from road import plot_road_comparison
        comparison = plot_road_comparison(r, directory)
        # Keep one plot window per scenario; road comparison is the main demo view.
        plt.close(fig)
        return comparison
    return fig


def animate(r, directory, gif=False):
    if 'roadside_readings' in r:
        from road import animate_road
        return animate_road(r, directory, gif)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylim(-3, 5)
    ax.set_xlim(-15, 90)
    ax.set_yticks([])
    ax.set_xlabel('Position relative to ego front bumper (m)')
    ax.axhspan(-1.6, 1.6, color='#dddddd')
    ax.axhline(0, color='white', linestyle='--')
    ego = Rectangle((-4.5, -.8), 4.5, 1.6, color='tab:blue')
    lead = Rectangle((65, -.8), 4.5, 1.6, color='tab:green')
    ax.add_patch(ego)
    ax.add_patch(lead)
    estimated, = ax.plot([], [], '|', markersize=35, color='tab:orange', label='Estimated lead rear bumper')
    measured, = ax.plot([], [], 'rx', label='Measured lead rear bumper')
    info = ax.text(.02, .95, '', transform=ax.transAxes, va='top', family='monospace')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f"{r['config'].scenario}: blue = ego, green = lead | illustrative warnings")
    # Render at most 20 frames/s; use frame time for correct playback speed.
    stride = max(1, int(np.ceil(.05 / r['config'].dt)))
    frames = list(range(0, len(r['t']), stride))
    def update(k):
        gap, speed = r['estimate'][k]
        lead.set_x(r['truth'][k, 0])
        estimated.set_data([gap], [0])
        measured.set_data([r['measurement'][k]], [0])
        ttc = r['ttc'][k]
        status = 'GAP CLOSED' if gap <= 0 else 'WARNING' if ttc < 3 else 'CAUTION' if ttc < 5 else 'MONITORING'
        lead.set_color('tab:red' if ttc < 3 else 'tab:orange' if ttc < 5 else 'tab:green')
        text_ttc = f'{ttc:.1f} s' if np.isfinite(ttc) else 'not closing'
        info.set_text(f"t={r['t'][k]:5.1f} s   gap={gap:6.1f} m   relative speed={speed:5.1f} m/s\nTTC={text_ttc}   {status}   sensor={'missing' if not np.isfinite(r['measurement'][k]) else 'available'}")
        ax.set_xlim(-15, max(90, gap + 15, r['truth'][k, 0] + 15))
        return lead, estimated, measured, info
    anim = FuncAnimation(fig, update, frames=frames, interval=1000*r['config'].dt*stride, blit=False, repeat=True)
    if gif:
        anim.save(directory / 'simulation.gif', writer=PillowWriter(fps=1/(r['config'].dt*stride)))
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, default='following')
    parser.add_argument('--all', action='store_true', help='Run all scenarios, including original experiments')
    parser.add_argument('--demo', action='store_true', help='Run both moving-road demonstrations')
    parser.add_argument('--headless', action='store_true', help='Save results without opening windows')
    parser.add_argument('--gif', action='store_true', help='Also save an animated GIF')
    parser.add_argument('--trials', type=int, default=1, help='Independent noise seeds per scenario')
    parser.add_argument('--duration', type=float, default=25.)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--sensor-std', type=float, default=2., help='Distance noise standard deviation (m); noisy scenario uses 3x')
    parser.add_argument('--accel-std', type=float, default=2., help='Process relative acceleration standard deviation (m/s²)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if args.trials < 1 or args.seed < 0:
        parser.error('Trials must be positive and seed nonnegative')
    import matplotlib
    if args.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, animations = [], []
    scenarios = SCENARIOS if args.all else ('following', 'obstacle') if args.demo else [args.scenario]
    for scenario in scenarios:
        for trial in range(args.trials):
            config = Config(scenario, args.duration, args.dt, args.sensor_std, args.accel_std, args.seed + trial)
            try:
                r = run(config)
            except ValueError as exc:
                parser.error(str(exc))
            directory = args.output / scenario
            if args.trials > 1:
                directory /= f'seed_{config.seed}'
            save_results(r, directory)
            row = {'scenario': scenario, 'seed': config.seed, **metrics(r)}
            rows.append(row)
            print(f"{scenario:8s} seed={config.seed}: distance RMSE sensor={row['distance_sensor_rmse_m']:.3f} m, KF={row['distance_kf_rmse_m']:.3f} m; speed KF={row['relative_speed_kf_rmse_mps']:.3f} m/s")
            if trial == 0:
                fig = plot_results(r, directory)
                if args.headless:
                    plt.close(fig)
                if args.gif or not args.headless:
                    fig, animation = animate(r, directory, args.gif)
                    animations.append(animation)
                    if args.headless:
                        plt.close(fig)
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results saved to {args.output.resolve()}')
    if not args.headless:
        plt.show()


if __name__ == '__main__':
    main()
