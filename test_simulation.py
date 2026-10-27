import unittest
import numpy as np
from simulation import Config, KalmanFilter, run, time_to_collision, SCENARIOS


class SimulationTests(unittest.TestCase):
    def test_exact_constant_velocity_is_recovered(self):
        kf = KalmanFilter(.1, .2, .1)
        for k in range(300):
            x, p, _ = kf.step(100 - 2*k*.1)
        np.testing.assert_allclose(x, [40.2, -2], atol=.01)

    def test_missing_measurement_is_prediction_only(self):
        kf = KalmanFilter(.1, 2, 2)
        kf.step(30)
        kf.step(29)
        x, p = kf.x.copy(), kf.P.copy()
        estimate, covariance, residual = kf.step(np.nan)
        np.testing.assert_allclose(estimate, kf.F @ x)
        np.testing.assert_allclose(covariance, kf.F @ p @ kf.F.T + kf.Q)
        self.assertTrue(np.isnan(residual))

    def test_scenarios_are_reproducible_and_covariance_valid(self):
        for scenario in SCENARIOS:
            r = run(Config(scenario=scenario))
            np.testing.assert_equal(r['measurement'], run(Config(scenario=scenario))['measurement'])
            self.assertTrue(np.isfinite(r['estimate']).all())
            np.testing.assert_allclose(r['covariance'], r['covariance'].transpose(0, 2, 1), atol=1e-12)
            self.assertTrue((np.linalg.eigvalsh(r['covariance']) >= -1e-12).all())
            self.assertTrue((r['truth'][:, 0] > 0).all())

    def test_ttc_sign_and_closed_gap(self):
        np.testing.assert_equal(time_to_collision([20, 20, 20, 0], [-4, 0, 2, -4]), [5, np.inf, np.inf, 0])

    def test_invalid_parameters(self):
        for config in (Config(dt=0), Config(sensor_std=-1), Config(duration=float('nan'))):
            with self.assertRaises(ValueError):
                run(config)

    def test_known_acceleration_input(self):
        kf = KalmanFilter(.1, 2, 2)
        x, _, _ = kf.step(30.)
        estimate, _, _ = kf.step(np.nan, relative_acceleration=4.)
        np.testing.assert_allclose(estimate, kf.F @ x + kf.G * 4.)

    def test_road_demos_brake_without_collision_across_noise_seeds(self):
        for scenario in ('following', 'obstacle'):
            for seed in range(42, 62):
                r = run(Config(scenario=scenario, seed=seed))
                self.assertGreater(r['truth'][:, 0].min(), 0)
                self.assertTrue((r['acceleration'] < 0).any())
                self.assertTrue(np.isfinite(r['roadside_readings']).any())
                self.assertTrue((r['ego_speed'] >= 0).all())
                if scenario == 'obstacle':
                    np.testing.assert_equal(r['lead'], np.full(len(r['t']), 100.))
                    self.assertEqual(r['ego_speed'][-1], 0.)


if __name__ == '__main__':
    unittest.main()
