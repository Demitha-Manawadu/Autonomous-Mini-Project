"""Generate the illustrated PDF and editable Word report from reproducible runs.

Run from the repository root: python report/build_report.py
Cover metadata can be edited in report/cover_details.json before rebuilding.
"""
import sys
import json
from pathlib import Path
from html import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation import Config, run, metrics
from road import animate_road

OUT = ROOT / 'report'
ASSETS = OUT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
META = OUT / 'cover_details.json'
if not META.exists():
    META.write_text(json.dumps({'name': 'Manawadu M.D', 'index': '220381M',
        'module': '[Module name and code]', 'department': '[Department name]',
        'university': 'University of Moratuwa', 'date': '27 October 2026'}, indent=2))
cover = json.loads(META.read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def generate_assets():
    results = {s: run(Config(scenario=s)) for s in ('following', 'obstacle', 'dropout')}
    trials = {s: [run(Config(scenario=s, seed=k)) for k in range(42, 62)] for s in ('following', 'obstacle')}
    summaries = {}
    for s, runs in trials.items():
        scores = [metrics(r) for r in runs]
        summaries[s] = {k: {'mean': float(np.mean([m[k] for m in scores])),
                             'std': float(np.std([m[k] for m in scores], ddof=1))}
                        for k in scores[0]}
        summaries[s]['min_true_gap_m'] = min(float(r['truth'][:, 0].min()) for r in runs)
        summaries[s]['all_stopped'] = all(r['ego_speed'][-1] == 0 for r in runs)
        r = results[s]
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 5.1), constrained_layout=True)
        t = r['t']
        axes[0].plot(t, r['measurement'], '.', color='#999999', alpha=.55, markersize=3, label='Noisy sensor')
        axes[0].plot(t, r['truth'][:, 0], 'k--', label='True gap')
        axes[0].plot(t, r['estimate'][:, 0], color='#126582', label='KF estimate')
        axes[0].set_ylabel('(a) Gap (m)')
        axes[1].plot(t, r['truth'][:, 1], 'k--', label='True relative speed')
        axes[1].plot(t, r['estimate'][:, 1], color='#126582', label='KF estimate')
        axes[1].set_ylabel('(b) Relative speed (m/s)')
        axes[2].plot(t, r['ego_speed'], color='#126582', label='Ego speed')
        axes[2].plot(t, r['lead_speed'], color='#bf862e', label='Target speed')
        axes[2].fill_between(t, 0, 19, where=r['acceleration'] < 0, color='red', alpha=.12, label='Braking command')
        axes[2].set(xlabel='Time (s)', ylabel='(c) Speed (m/s)')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='upper right')
        fig.savefig(ASSETS / f'{s}_results.png', dpi=180)
        plt.close(fig)
        # Reuse the actual animation renderer for a road snapshot.
        fig, animation = animate_road(r, ASSETS)
        animation._init_draw()
        animation._func(100 if s == 'following' else 140)
        fig.savefig(ASSETS / f'{s}_road.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), constrained_layout=True)
    for ax, keys, ylabel in zip(axes,
         [('distance_sensor_rmse_m', 'distance_kf_rmse_m'),
          ('relative_speed_difference_rmse_mps', 'relative_speed_kf_rmse_mps')],
         ['Distance RMSE (m)', 'Relative-speed RMSE (m/s)']):
        x = np.arange(2)
        for j, (key, label, color) in enumerate(zip(keys, ['Raw baseline', 'KF'], ['#b5b8bd', '#126582'])):
            means = [summaries[s][key]['mean'] for s in trials]
            stds = [summaries[s][key]['std'] for s in trials]
            ax.bar(x+(j-.5)*.34, means, .34, yerr=stds, capsize=3, color=color, label=label)
        ax.set(xticks=x, xticklabels=['Following', 'Obstacle'], ylabel=ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(ASSETS/'rmse_summary.png', dpi=180)
    plt.close(fig)
    r = results['dropout']
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 3.8), constrained_layout=True)
    sigma = np.sqrt(r['covariance'][:, 0, 0])
    axes[0].plot(r['t'], r['truth'][:, 0], 'k--', label='True gap')
    axes[0].plot(r['t'], r['estimate'][:, 0], color='#126582', label='KF estimate')
    axes[0].fill_between(r['t'], r['estimate'][:, 0]-1.96*sigma, r['estimate'][:, 0]+1.96*sigma, alpha=.2, label='Nominal 95% interval')
    axes[0].set_ylabel('Gap (m)')
    axes[1].plot(r['t'], sigma, color='#126582', label='Distance standard deviation')
    axes[1].set(xlabel='Time (s)', ylabel='Uncertainty (m)')
    for ax in axes:
        ax.axvspan(8, 10, alpha=.15, color='gray')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(ASSETS/'dropout.png', dpi=180)
    plt.close(fig)
    tuning = []
    for a in (.5, 2., 4., 8.):
        runs = [metrics(run(Config(scenario='braking', accel_std=a, seed=k))) for k in range(42,62)]
        tuning.append([a, np.mean([m['distance_kf_rmse_m'] for m in runs]), np.mean([m['relative_speed_kf_rmse_mps'] for m in runs])])
    evidence = {'road_20_seed_summary': summaries, 'braking_tuning': np.asarray(tuning).tolist(),
                'seed42': {s: metrics(r) for s,r in results.items()}}
    (OUT/'report_evidence.json').write_text(json.dumps(evidence, indent=2))
    return results, summaries, tuning


def content(results, summary, tuning):
    pages = []
    def page(title, blocks):
        pages.append((title, blocks))
    P = lambda t: ('p', t)
    H = lambda t: ('h', t)
    C = lambda t: ('code', t)
    I = lambda name, cap, width=16: ('image', name, cap, width)
    T = lambda rows, cap: ('table', rows, cap)
    page('Abstract and report guide', [
        H('Abstract'),
        P('This project estimates the distance and relative speed between a vehicle and an object ahead. A linear Kalman filter (KF) processes noisy distance readings. Two Python demonstrations are used: following a braking vehicle and approaching a stationary obstacle. A simple controller uses the estimates to command braking.'),
        P('The filter is implemented directly with NumPy. Across 20 noise seeds, mean distance error falls from 2.007 m to 1.206 m in the following scenario and to 0.697 m in the obstacle scenario. Relative-speed estimates are also much less noisy than direct distance differences. The simulation shows the value of filtering, while braking transients and sensor loss reveal its limits.'),
        H('Report guide'),
        T([['Section', 'Content'], ['1', 'Problem, objectives and sensor arrangement'], ['2', 'Motion model and noise assumptions'], ['3', 'Kalman filter equations'], ['4', 'Python implementation and animation'], ['5', 'Following a braking vehicle'], ['6', 'Approaching a stationary obstacle'], ['7', 'Repeated-trial evaluation'], ['8', 'Parameter tuning and sensor loss'], ['9', 'Discussion and conclusion'], ['10', 'Running the project and references']], 'Report structure'),
        H('Project details'),
        P('Filter: linear Kalman filter. Application: longitudinal vehicle safety and target tracking. Platform: Python. Data: synthetic range measurements. The study tracks one in-lane target. It does not build a map or plan a steering path.'),
        P('The source code is available at the repository listed in Section 10. All tables and plots in this report are generated from the local implementation. No road tests or hardware sensor measurements are claimed.')])
    page('1. Problem and simulation arrangement', [
        H('1.1 Aim and objectives'),
        P('A distance sensor does not give a perfectly steady reading. Small changes in successive noisy readings can also produce a very poor speed estimate. The aim is to estimate the true gap and closing speed well enough to support a simple warning or braking decision.'),
        P('The objectives are to implement a two-state KF, simulate moving and stationary targets, compare estimated states with known truth, and study the effect of noise, braking and missing readings.'),
        H('1.2 Sensor, filter and controller'),
        C('Vehicle motion -> noisy range sensor -> KF estimate\n                                      -> risk check -> ego braking\nGround truth --------------------------> error evaluation only'),
        P('The simulated sensor measures the bumper-to-bumper gap to a known object in the lane. It has a forward range of 180 m. Zero-mean Gaussian noise is added to each valid reading. The default standard deviation is 2 m. The KF receives these readings, not the true target position or speed.'),
        H('1.3 What the road display means'),
        I('obstacle_road.png', 'Figure 1. Fixed-world road view in the stationary-obstacle scenario at 14 s. Ego has stopped short of the obstacle. The lower plot compares gap readings and estimates.', 15.5),
        P('The ego rectangle is blue and turns red during braking. The target is green. A cyan line represents the sensor beam. Red crosses show measured target position; the orange marker shows the KF estimate. Gray blocks are fixed roadside objects. Magenta rings mark visible roadside objects.'),
        P('Roadside ranges are also generated and saved. Object identity and lane membership are supplied by the simulator. Objects outside the lane do not enter the in-lane target filter. This simplifies detection; no camera recognition or multi-object association is implemented.')])
    page('2. Motion model and assumptions', [
        H('2.1 State and sign convention'),
        P('Let d be the gap in metres. Let v_rel = v_target - v_ego in m/s. A negative relative speed means the gap is closing. The state contains distance and relative speed.'),
        C('x = [d, v_rel]^T\nd(k+1) = d(k) + dt*v_rel(k) + 0.5*dt^2*u(k)\nv_rel(k+1) = v_rel(k) + dt*u(k)'),
        P('These equations follow from one-dimensional motion over a short interval. In the road scenarios, u is minus the known executed ego acceleration. Unknown target acceleration is represented by process uncertainty. The sensor model is linear, so an Extended KF is not required. Reference [1] describes the same class of linear tracking model.'),
        H('2.2 State-space form'),
        C('x(k+1) = F*x(k) + G*u(k) + w(k)\nz(k)   = H*x(k) + n(k)\n\nF = [[1, dt], [0, 1]]    G = [dt^2/2, dt]^T\nH = [[1, 0]]             u = -a_ego'),
        P('With a positive time step, the two rows H and HF form an observability matrix with determinant dt. It therefore has rank two. This explains why relative speed can be estimated from a sequence of distance observations even when the sensor does not measure speed.'),
        H('2.3 Noise and initial uncertainty'),
        C('Q = sigma_a^2 * G*G^T     R = sigma_z^2\nx(0) = [first range reading, 0]^T\nP(0) = diag(R, 100)'),
        P('At dt = 0.1 s, sigma_a = 2 m/s² and sigma_z = 2 m, Q = [[0.0001, 0.002], [0.002, 0.04]] and R = 4 m². The process model assumes an independent random relative acceleration held constant within each sample interval. Its tuning depends on the sampling interval.'),
        P('Initial relative speed is unknown and is set to zero with a large variance of 100 (m/s)². The first observation initializes distance and is not applied again as a correction. This avoids counting the same measurement twice.')])
    page('3. Kalman filter calculation', [
        H('3.1 Predict'),
        C('x_pred = F*x + G*u\nP_pred = F*P*F^T + Q'),
        P('Prediction moves the estimated state forward by one sample. The known ego acceleration accounts for the motion of the observer. Covariance P describes uncertainty in the state. Adding Q allows the estimate to change when the target does not follow the expected motion.'),
        H('3.2 Correct with the sensor reading'),
        C('innovation y = z - H*x_pred\nS = H*P_pred*H^T + R\nK = P_pred*H^T / S\nx = x_pred + K*y'),
        P('The innovation is the difference between the observed and predicted distance. The Kalman gain K sets how much that difference changes distance and relative speed. A larger R gives less weight to a reading. Greater predicted uncertainty generally gives it more weight.'),
        H('3.3 Update uncertainty'),
        C('A = I - K*H\nP = A*P_pred*A^T + K*R*K^T'),
        P('This is the Joseph covariance update. It is used to reduce numerical problems with covariance symmetry and non-negative uncertainty. When a reading is missing, the correction is skipped. Only the predicted state and covariance are kept.'),
        H('3.4 Time to collision and braking'),
        C('TTC = d / (-v_rel), if d > 0 and v_rel < -0.1 m/s\nTTC = infinity, if not closing; TTC = 0 if gap <= 0'),
        P('For example, a 20 m gap closing at 5 m/s gives a TTC of 4 s. The road controller commands -4 m/s² when estimated TTC is below 5 s and estimated closing speed exceeds 0.3 m/s. In the known stationary-obstacle case, an estimated gap below 12 m also maintains braking. Ego speed cannot become negative.'),
        P('TTC is calculated after the KF update. It assumes relative speed stays unchanged. The controller is a small demonstration of how estimates can be used; it is not a validated emergency-braking system.')])
    page('4. Python implementation', [
        H('4.1 Files and libraries'),
        T([['Part', 'Purpose'], ['simulation.py / NumPy', 'KF matrices, prediction, correction, TTC and errors'], ['road.py / NumPy', 'Vehicle motion, noisy sensor and brake response'], ['road.py / Matplotlib', 'Road rectangles and FuncAnimation playback'], ['main.py', 'Commands, plots and CSV/JSON export'], ['Pillow', 'Optional animated GIF export'], ['test_simulation.py', 'Seven automated checks']], 'Table 1. Main implementation components.'),
        H('4.2 Generating a moving object'),
        C('ego_speed[k+1] = max(0., ego_speed[k] + acceleration[k]*dt)\nego[k+1] = ego[k] + 0.5*(ego_speed[k] + ego_speed[k+1])*dt\n\ngap = lead[k] - ego[k]\nmeasurement[k] = gap + noise[k]'),
        P('This excerpt follows the road simulation code, using dt for c.dt. Speed is updated first. Position is then advanced using average speed. At 18 m/s, a vehicle travels about 1.8 m in one 0.1 s step. A stationary obstacle keeps the same position. The ego position marks its front bumper, while target position marks its rear bumper.'),
        H('4.3 Connecting the KF and animation'),
        C('actual_ego_accel = (ego_speed[k] - ego_speed[k-1]) / dt\nestimate[k], covariance[k], innovation[k] = (\n    kf.step(measurement[k], -actual_ego_accel)\n)\n# Animation places rectangles at the stored positions:\nego_rectangle.set_x(ego_position - 4.5)'),
        P('The acceleration expression applies after the first sample; the initial value is zero. In this ideal simulator executed ego acceleration is available without error. The filter never receives true target speed or true gap.'),
        P('The full trajectory is calculated before animation starts. Each sample still follows a causal sensor-update-control sequence. Matplotlib then plays those stored states at approximately real-time speed. Replaying the animation does not run new sensor samples. A different random seed produces another noise realization.'),
        P('Running this two-state KF is inexpensive. A local timing check averaged about 17 microseconds per update over 10,000 updates, excluding graphics. This is an illustrative timing, not a portable benchmark. The challenging work is modelling, tuning and validation, rather than the small matrix calculation.')])
    page('5. Scenario A: following a braking vehicle', [
        P('Both vehicles start at 18 m/s with a 50 m gap. The lead vehicle brakes at -3 m/s² from 4 to 8 s, reaching 6 m/s. Ego has no scripted braking time: its controller uses the estimated gap and closing speed. It coasts or brakes, but does not accelerate again.'),
        I('following_results.png', 'Figure 2. Following scenario, seed 42. (a) Distance measurements and estimates. (b) True and estimated relative speed. (c) Vehicle speeds; shaded intervals show commanded braking.', 15.8),
        P('Before braking, the gap is nearly constant. When the lead slows, relative speed becomes negative and the gap falls. The KF needs several observations to identify this change, so its estimate briefly lags. Ego then brakes and the closing speed decreases.'),
        P(f"For this run, sensor distance RMSE is {metrics(results['following'])['distance_sensor_rmse_m']:.3f} m and KF distance RMSE is {metrics(results['following'])['distance_kf_rmse_m']:.3f} m. The true minimum gap is {results['following']['truth'][:,0].min():.2f} m. The controller is brake-only, so this is not a complete adaptive cruise-control system.")])
    page('6. Scenario B: stationary obstacle', [
        P('Ego starts at 18 m/s. A stationary in-lane obstacle is placed at 100 m. Its true speed is zero throughout the run. Roadside objects are placed at longitudinal positions 30, 75, 125 and 180 m, with lateral offsets of ±5 m.'),
        I('obstacle_results.png', 'Figure 3. Stationary-obstacle scenario, seed 42. (a) Gap to the obstacle. (b) Relative speed. (c) Ego braking to rest; the target speed remains zero.', 15.8),
        P('Initially, the true relative speed is -18 m/s, but the KF starts at zero because it has only one distance reading. Later readings reveal the closing speed. As braking reduces ego speed, relative speed approaches zero and the gap settles at a positive value.'),
        P(f"For seed 42, ego stops with a true gap of {results['obstacle']['truth'][-1,0]:.2f} m. Sensor distance RMSE is {metrics(results['obstacle'])['distance_sensor_rmse_m']:.3f} m and KF distance RMSE is {metrics(results['obstacle'])['distance_kf_rmse_m']:.3f} m. Small fluctuations remain even after stopping because sensor readings remain noisy.")])
    rows = [['Scenario', 'Raw distance\nRMSE (m)', 'KF distance\nRMSE (m)', 'Raw speed\nRMSE (m/s)', 'KF speed\nRMSE (m/s)']]
    for s in summary:
        rows.append([s.capitalize()] + [f"{summary[s][k]['mean']:.3f}" for k in ['distance_sensor_rmse_m','distance_kf_rmse_m','relative_speed_difference_rmse_mps','relative_speed_kf_rmse_mps']])
    page('7. Evaluation across repeated trials', [
        P('Each road scenario is repeated with seeds 42 to 61. Each run has 251 samples over 25 s at 10 Hz. The sensor standard deviation is 2 m and process acceleration standard deviation is 2 m/s². Noise changes between seeds, and the closed-loop vehicle response may change too.'),
        C('RMSE = sqrt(mean((estimate - ground_truth)^2))\nRaw speed[k] = (measurement[k] - measurement[k-1]) / dt'),
        T(rows, 'Table 2. Arithmetic mean of per-run RMSE over 20 seeds. Initialisation transients are included. Raw speed uses valid successive readings only.'),
        I('rmse_summary.png', 'Figure 4. Mean RMSE across 20 seeds. Error bars show one sample standard deviation of per-run RMSE, not a confidence interval.', 16),
        P('Mean range RMSE is reduced by about 40% for following and 65% for the stationary obstacle. Raw distance differences are much noisier than the KF speed estimate. Independent distance noise gives a difference-noise standard deviation of about sqrt(2) × 2 / 0.1 = 28.3 m/s.'),
        P('Both estimators are evaluated on the same driven trajectory in each trial. A separate vehicle controlled by raw readings is not simulated. The results therefore show an estimation benefit, not a measured improvement in collision rate over another controller.'),
        P(f"All 40 default road runs keep a positive gap. The smallest gaps across the tested seeds are {summary['following']['min_true_gap_m']:.2f} m for following and {summary['obstacle']['min_true_gap_m']:.2f} m for the obstacle. All 20 obstacle runs end with ego stopped. These outcomes apply to the tested conditions only.")])
    page('8. Tuning and temporary sensor loss', [
        P('A supporting experiment uses the original scripted braking trajectory. Keeping that trajectory fixed helps separate filter tuning from changes caused by the feedback controller. Sensor noise stays at 2 m. The following values are means across seeds 42–61.'),
        T([['Process sigma_a\n(m/s²)', 'KF distance\nRMSE (m)', 'KF speed\nRMSE (m/s)']] + [[f'{a:g}', f'{d:.3f}', f'{v:.3f}'] for a,d,v in tuning], 'Table 3. Effect of process uncertainty on the same open-loop braking experiment.'),
        P('A small process uncertainty strongly favours the predicted motion and can delay the response to braking. A larger value permits faster changes but can pass more noise. The values above are specific to this sampling interval and motion; there is no single setting that is best for every trajectory.'),
        I('dropout.png', 'Figure 5. Supporting sensor-loss experiment, seed 42. The gray band marks missing measurements from 8 s to just before 10 s. Uncertainty grows during prediction-only steps.', 15.8),
        P('During the outage, the KF continues to predict but cannot correct its estimate. This outage overlaps braking, so the state can drift. For seed 42, full-run KF distance RMSE is 2.095 m. Raw range RMSE is 1.901 m on available samples only; the two values use different sample sets. Matched-sample errors are also exported by the program.'),
        P('The shaded interval is a nominal Gaussian 95% interval. It is not guaranteed to contain truth 95% of the time when the motion model is wrong. The original 20-seed braking experiment gives about 83% distance coverage, indicating under-estimated transient uncertainty.')])
    page('9. Discussion and conclusion', [
        H('9.1 What the KF adds'),
        P('The sensor gives distance observations. The KF combines the current observation with a prediction based on previous observations. It reduces range noise and estimates a hidden quantity, relative speed. It can also continue predicting briefly when observations are missing. These are useful inputs to a safety decision.'),
        P('The filter does not identify an obstacle by itself. Detection, state estimation and control are separate tasks. Here the simulator supplies the target identity and lane label. One KF follows the in-lane target. Roadside detections are logged but are not fused into that target track.'),
        H('9.2 Limits of this study'),
        P('Motion is one-dimensional. There are no curves, steering decisions, intersections or maps. Sensor noise is Gaussian with known variance. There are no clutter returns, incorrect lane labels or changing object identities. The controller assumes ideal braking with no actuator delay, tyre model or road-friction limit.'),
        P('The KF assumes near-constant relative velocity between updates, apart from known ego acceleration. Strong target braking causes lag. At start-up, speed is uncertain. During long outages, prediction can become inaccurate. TTC assumes unchanged relative speed and is sensitive near zero closing speed.'),
        H('9.3 Verification'),
        P('Seven automated tests check constant-speed convergence, prediction without measurements, repeatability and valid covariance, TTC signs, invalid parameters, known acceleration input, and road outcomes across 20 seeds per scenario. All pass. Frames from both road animations and GIF exports have also been checked.'),
        H('9.4 Conclusion'),
        P('A two-state linear KF is suitable for this simplified longitudinal tracking problem. The measured results show lower distance error and much lower speed noise than simple raw baselines. The road demonstrations explain how those estimates can support braking. The tuning and dropout experiments show why assumptions and uncertainty still matter.'),
        P('Further work could use real range recordings, test incorrect noise settings, add acceleration to the state, and compare controllers under the same disturbances. Multiple objects would require separate tracks and measurement association. Steering avoidance would also require lateral motion and path planning.')])
    page('10. Reproduction and references', [
        H('10.1 Run the demonstrations'),
        C('python -m pip install -r requirements.txt\npython main.py --demo\n\n# Individual scenarios\npython main.py --scenario following\npython main.py --scenario obstacle\n\n# Repeated trials and verification\npython main.py --demo --headless --trials 20 --output outputs/road_trials\npython -m unittest -v'),
        P('The demonstration command calculates fresh trajectories and then opens the comparison plots and repeating animations. Seed 42 is the default. Add --seed 73, for example, to change the measurement noise. Add --headless to save plots without opening windows, or --gif to export animation.'),
        P('results.csv contains truth, measurements, estimated states, uncertainty and TTC. road_sensor.csv contains world positions, vehicle speeds, brake commands and roadside ranges. metrics.json and summary.csv contain numerical results. report/report_evidence.json stores the exact repeated-trial and tuning values used for this report.'),
        H('10.2 Source code'),
        P('Repository: https://github.com/Demitha-Manawadu/Autonomous-Mini-Project.git'),
        P('The report describes the local Python implementation. The full code, run instructions and test files should accompany the submission. The report builder is report/build_report.py; cover details can be edited in report/cover_details.json.'),
        H('10.3 References'),
        P('[1] MathWorks. “trackingKF — Linear Kalman filter for object tracking.” Used as a reference for the linear tracking model and optional control input. https://www.mathworks.com/help/driving/ref/trackingkf.html'),
        P('[2] University of Moratuwa. “The Emblem of University.” Source of the cover emblem. https://www.uom.lk/emblem-university'),
        P('[3] Mini Project: Implementing State Estimation Filters. Assignment guideline supplied with the project, two pages; no module code or publication date stated.'),
        P('[4] Manawadu M.D. Autonomous-Mini-Project. Python source repository, link above. All reported experimental results and figures were generated from this implementation.'),
        P('Web references checked on 27 October 2026. No external data set was used.')])
    return pages


def export(pages):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/times.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/timesbd.ttf'))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    navy, gold = colors.black, colors.black
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport',fontName='Arial',fontSize=10.2,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#242c33')))
    styles.add(ParagraphStyle(name='TitleReport',fontName='Arial-Bold',fontSize=18,leading=23,spaceAfter=16,textColor=navy))
    styles.add(ParagraphStyle(name='HeadingReport',fontName='Arial-Bold',fontSize=11.5,leading=15,spaceBefore=6,spaceAfter=6,textColor=navy))
    styles.add(ParagraphStyle(name='CaptionReport',fontName='Arial',fontSize=9,leading=12,spaceAfter=10,textColor=colors.HexColor('#505c66')))
    styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=8.3,leading=11,backColor=colors.HexColor('#f0f3f5'),borderPadding=7,spaceBefore=4,spaceAfter=12))
    styles.add(ParagraphStyle(name='CellReport',fontName='Arial',fontSize=8.7,leading=12))
    story = []
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21)
    sec.top_margin,sec.bottom_margin,sec.left_margin,sec.right_margin = Cm(2),Cm(2),Cm(2.3),Cm(2.3)
    normal = doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.2)
    normal.paragraph_format.space_after=Pt(8)
    normal.paragraph_format.line_spacing=1.12
    for name in ('Title','Heading 1','Heading 2'):
        doc.styles[name].font.name='Times New Roman';doc.styles[name].font.color.rgb=RGBColor.from_string('000000')
    border = OxmlElement('w:pgBorders');border.set(qn('w:offsetFrom'),'page')
    for edge in ('top','left','bottom','right'):
        el=OxmlElement('w:'+edge)
        for k,v in [('val','single'),('sz','5'),('space','18'),('color','000000')]:el.set(qn('w:'+k),v)
        border.append(el)
    sec._sectPr.append(border)
    sec.different_first_page_header_footer=True
    header=sec.header.paragraphs[0];header.text='EN4594 — Autonomous Systems'
    header.runs[0].font.size=Pt(8)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
    def safe(t):
        return escape(t).replace('\n','<br/>')
    def body(t,kind='BodyReport'):
        # Keep repository and external references clickable in PDF.
        import re
        t=safe(t)
        t=re.sub(r'(https://[^\s<]+)',r'<link href="\1" color="#126582">\1</link>',t)
        return Paragraph(t, styles[kind])
    logo=ASSETS/'uom_emblem.jpg'
    if not logo.exists():
        raise FileNotFoundError('Official emblem is missing: report/assets/uom_emblem.jpg')
    story.append(Spacer(1,22))
    story.append(Image(str(logo),width=4.2*cm,height=4.8*cm,kind='proportional'))
    story.append(Spacer(1,12))
    center=ParagraphStyle('cover',parent=styles['BodyReport'],alignment=TA_CENTER,fontSize=14,leading=19,spaceAfter=11,textColor=colors.black)
    ct=ParagraphStyle('covertitle',parent=styles['TitleReport'],alignment=TA_CENTER,fontSize=20,leading=25,spaceBefore=14,spaceAfter=18)
    module_style=ParagraphStyle('covermodule',parent=ct,fontSize=15,leading=19,spaceBefore=12,spaceAfter=8)
    mini_style=ParagraphStyle('covermini',parent=ct,fontSize=24,leading=29,spaceBefore=12,spaceAfter=14)
    small=ParagraphStyle('coversmall',parent=center,fontSize=12,leading=16,spaceAfter=8)
    repo_style=ParagraphStyle('coverrepo',parent=small,fontSize=10,leading=14)
    from latex_report import TITLE
    cover_lines=[(cover['university'],center), (cover.get('faculty','Faculty of Engineering'),center), (cover['department'],center),
        ('EN4594',module_style),('Autonomous Systems',module_style),('Mini Project Report',mini_style),
        (TITLE,ct),
        (cover['name'],center),('Index Number: '+cover['index'],center),(cover['date'],small),
        ('Source Code Repository',center),('https://github.com/Demitha-Manawadu/Autonomous-Mini-Project.git',repo_style)]
    for text,style in cover_lines:
        if text.startswith('https://'):
            story.append(Paragraph('<link href="'+text+'">'+safe(text)+'</link>',style))
        else:story.append(Paragraph(safe(text),style))
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(logo),height=Cm(4.8))
    for text,style in cover_lines:
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(10)
        r=p.add_run(text);r.font.size=Pt(24 if style==mini_style else 20 if style==ct else 15 if style==module_style else 10 if style==repo_style else 14);r.bold=style in (ct,mini_style)
    doc.add_page_break()
    doc.add_heading('Table of Contents',level=1)
    toc=OxmlElement('w:fldSimple');toc.set(qn('w:instr'),'TOC \\o "1-2" \\h \\z \\u')
    doc.add_paragraph()._p.append(toc)
    for section,(title,blocks) in enumerate(pages,1):
        story.append(PageBreak());story.append(body(title,'TitleReport'))
        doc.add_page_break();doc.add_heading(f'{section}. {title}',level=1)
        for b in blocks:
            if b[0] in ('p','h','subh'):
                story.append(body(b[1], 'HeadingReport' if b[0] in ('h','subh') else 'BodyReport'))
                if b[0] in ('h','subh'):doc.add_heading(b[1],level=2 if b[0]=='h' else 3)
                else:doc.add_paragraph(b[1])
            elif b[0]=='break':
                story.append(PageBreak())
                doc.add_page_break()
            elif b[0]=='code':
                story.append(Preformatted(b[1],styles['CodeReport']))
                p=doc.add_paragraph();p.paragraph_format.space_after=Pt(9)
                r=p.add_run(b[1]);r.font.name='Consolas';r.font.size=Pt(8)
            elif b[0]=='equation':
                story.append(body(b[2]))
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                r=p.add_run(b[2]);r.font.name='Cambria Math';r.font.size=Pt(10)
            elif b[0]=='image':
                path=ASSETS/b[1]
                w,h=PILImage.open(path).size
                width=b[3]*cm
                story.append(Image(str(path),width=width,height=width*h/w))
                story.append(body(b[2],'CaptionReport'))
                doc.add_picture(str(path),width=Cm(b[3]))
                p=doc.add_paragraph(b[2]);p.style='Caption'
            elif b[0]=='table':
                rows=[[Paragraph(safe(str(cell)),styles['CellReport']) for cell in row] for row in b[1]]
                table=Table(rows,colWidths=[16.4*cm/len(rows[0])]*len(rows[0]),hAlign='LEFT',repeatRows=1)
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1eaf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#bac7d0')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story.append(table);story.append(Spacer(1,6));story.append(body(b[2],'CaptionReport'))
                table=doc.add_table(rows=0,cols=len(b[1][0]));table.style='Light Shading Accent 1'
                for row in b[1]:
                    cells=table.add_row().cells
                    for cell,text in zip(cells,row):cell.text=str(text)
                doc.add_paragraph(b[2],style='Caption')
    def decorate(canvas,document):
        canvas.saveState();w,h=document.pagesize
        canvas.setStrokeColor(gold);canvas.setLineWidth(.6);canvas.rect(22,22,w-44,h-44)
        if document.page!=1:
            canvas.setFont('Arial',8);canvas.setFillColor(navy)
            canvas.drawString(2.3*cm,h-1.35*cm,'STATE ESTIMATION MINI PROJECT')
            canvas.drawRightString(w-2.3*cm,1.1*cm,str(document.page))
        canvas.restoreState()
    pdf=OUT/'Manawadu_220381M_KF_Report.pdf'
    SimpleDocTemplate(str(pdf),pagesize=(21*cm,29.7*cm),leftMargin=2.3*cm,rightMargin=2.3*cm,topMargin=2*cm,bottomMargin=1.8*cm,title='Distance and Relative Speed Estimation for Autonomous Vehicle Safety',author=cover['name']).build(story,onFirstPage=decorate,onLaterPages=decorate)
    doc.core_properties.author=cover['name'];doc.core_properties.title='Distance and Relative Speed Estimation for Autonomous Vehicle Safety'
    doc.save(OUT/'Manawadu_220381M_KF_Report.docx')
    print('Created PDF and Word reports in', OUT)


if __name__=='__main__':
    r,s,t=generate_assets()
    from latex_report import revise,export_latex
    pages=revise(content(r,s,t),ASSETS)
    export(pages)
    export_latex(pages,OUT,cover)
