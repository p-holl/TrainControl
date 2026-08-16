import unittest
from dataclasses import replace

from fpme.model import DrivingModel, TrainTracker


class TestTrainTrackerIntegrateTo(unittest.TestCase):

    def setUp(self):
        self.model = DrivingModel(
            key_speeds=(0., 20., 40., 60., 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999),
            acceleration=10.,   # cm/s^2
            deceleration=10.,   # cm/s^2
            deceleration_on_reverse=100.,
            sound_function=0,
            reverse_time_with_sound=2.,
            startup_time=1.,
        )

    def test_simple_acceleration_reaches_target(self):
        tracker = TrainTracker(model=self.model)
        tracker.set(speed_index=2, in_reverse=False, functions={}, current_time=0.)
        # target speed = 40 cm/s, accel = 10 cm/s^2 -> reaches target at t=4
        speed, sgn_d, abs_d = tracker.integrate_to(2.)
        self.assertAlmostEqual(speed, 20.0)
        self.assertAlmostEqual(sgn_d, 20.0)  # avg speed 10 * 2s
        self.assertAlmostEqual(abs_d, 20.0)

    def test_reversal_time_after_multiple_updates(self):
        tracker = TrainTracker(model=self.model)
        tracker.set(speed_index=0, in_reverse=False, functions={0: True}, current_time=0.)
        tracker.set(speed_index=5, in_reverse=True, functions={0: True}, current_time=1.)
        # Verify direction changed event occurred
        self.assertIsNotNone(tracker.direction_changed_at)
        self.assertEqual(tracker.direction_changed_at, 1.0)
        tracker.set(speed_index=6, in_reverse=True, functions={0: True}, current_time=1.0001)
        # Direction should not be marked as changed again (no direction change between 1.0 and 1.0001)
        self.assertEqual(tracker.direction_changed_at, 1.0)  # Still the original time
        # target speed = 40 cm/s, accel = 10 cm/s^2 -> reaches target at t=4
        speed, sgn_d, abs_d = tracker.integrate_to(2.)
        self.assertAlmostEqual(speed, 0)
        self.assertAlmostEqual(sgn_d, 0)  # avg speed 10 * 2s
        self.assertAlmostEqual(abs_d, 0)

    def test_acceleration_then_constant_speed(self):
        tracker = TrainTracker(model=self.model)
        tracker.set(speed_index=2, in_reverse=False, functions={}, current_time=0.)
        # target = 40, accel=10 -> reaches target at t=4s, then constant for 2s more
        speed, sgn_d, abs_d = tracker.integrate_to(6.)
        self.assertAlmostEqual(speed, 40.0)
        # distance during accel: avg 20 * 4 = 80; during constant: 40*2 = 80
        self.assertAlmostEqual(sgn_d, 160.0)
        self.assertAlmostEqual(abs_d, 160.0)

    def test_no_sound_direction_change_uses_deceleration_on_reverse(self):
        tracker = TrainTracker(model=self.model)
        tracker.set(speed_index=2, in_reverse=False, functions={}, current_time=0.)
        tracker.integrate_to(4.)  # reach target speed 40
        tracker.set(speed_index=2, in_reverse=False, functions={}, current_time=4.)
        # simulate having reached target speed by directly setting state
        tracker.speed = 40.
        tracker.sgn_distance = 0.
        tracker.abs_distance = 0.
        tracker.last_update = 4.
        tracker.set(speed_index=2, in_reverse=True, functions={}, current_time=4.)
        # direction_changed=True, no sound => reverse_time=0
        # decel_on_reverse=100 -> t_stop = 40/100 = 0.4s
        speed, sgn_d, abs_d = tracker.integrate_to(4.4)
        self.assertAlmostEqual(speed, 0.0, places=5)
        self.assertAlmostEqual(sgn_d, 8.0)  # avg 20 * 0.4
        self.assertAlmostEqual(abs_d, 8.0)

    def test_direction_change_with_sound_waits_reverse_time(self):
        tracker = TrainTracker(model=self.model)
        functions = {0: True}
        tracker.set(speed_index=2, in_reverse=False, functions=functions, current_time=0.)
        tracker.speed = 40.
        tracker.sgn_distance = 0.
        tracker.abs_distance = 0.
        tracker.last_update = 0.
        tracker.set(speed_index=2, in_reverse=True, functions=functions, current_time=0.)
        # t_stop = 0.4s, then reverse_time_with_sound = 2s wait
        speed, sgn_d, abs_d = tracker.integrate_to(1.0)
        self.assertAlmostEqual(speed, 0.0)
        self.assertAlmostEqual(sgn_d, 8.0)  # only distance from decel phase
        self.assertAlmostEqual(abs_d, 8.0)

        speed2, sgn_d2, abs_d2 = tracker.integrate_to(2.4)
        # 0.4 decel + 2.0 wait = 2.4s total elapsed, still at boundary, speed 0
        self.assertAlmostEqual(speed2, 0.0)
        self.assertAlmostEqual(sgn_d2, 8.0)

    def test_deceleration_toward_lower_target(self):
        tracker = TrainTracker(model=self.model)
        functions = {}
        tracker.set(speed_index=3, in_reverse=False, functions=functions, current_time=0.)
        tracker.speed = 60.
        tracker.sgn_distance = 0.
        tracker.abs_distance = 0.
        tracker.last_update = 0.
        tracker.set(speed_index=1, in_reverse=False, functions=functions, current_time=0.)
        # target=20, current=60, decel=10 -> t_reach=4s
        speed, sgn_d, abs_d = tracker.integrate_to(2.)
        self.assertAlmostEqual(speed, 40.0)
        self.assertAlmostEqual(sgn_d, 100.0)  # avg(60,40)*2
        self.assertAlmostEqual(abs_d, 100.0)


if __name__ == '__main__':
    unittest.main()