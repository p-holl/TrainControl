from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np


def speeds(s15, exponent=1.3):
    return (*np.linspace(0, s15 ** (1/exponent), 15) ** exponent,)


@dataclass(frozen=True)
class DrivingModel:
    key_speeds: Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float] = speeds(60, 1.3)  # cm/s
    acceleration: float = 60.
    deceleration_on_reverse: float = 1000.
    sound_function: Optional[int] = None  # function index. None if has no sound
    reverse_time_with_sound: float = 0.
    startup_time: float = 0.


@dataclass
class TrainTracker:
    model: DrivingModel
    # --- Last update ---
    last_update: float = -1  # perf_counter time
    speed_index: int = 0  # current signal [0, 15]
    direction_changed: float = -100.
    in_reverse: bool = False
    sound_switched_on: float = -100.  # this causes a wait of model.startup_time if the train was stopped.
    functions: Dict[int, bool] = field(default_factory=dict)
    # --- Model state ---
    speed: float = 0.  # Modeled signed speed (cm/s) (negative when reversed) at t=last_update
    sgn_distance: float = 0.  # distance traveled (cm) at t=last_update
    abs_distance: float = 0. # distance traveled (cm) at t=last_update

    def __post_init__(self):
        assert self.model.acceleration > 0
        assert self.model.deceleration_on_reverse > 0

    def set(self, speed_index: int, in_reverse: bool, functions: Dict[int, bool], current_time: float):
        # --- Integrate to now ---
        if self.last_update == -1:  # initialization event
            self.speed, self.sgn_distance, self.abs_distance = 0., 0., 0.
        else:
            self.speed, self.sgn_distance, self.abs_distance = self.integrate_to(current_time)
        # --- Update state ---
        if functions.get(self.model.sound_function, False) and not self.is_sound_on and self.speed_index == 0:
            self.sound_switched_on = current_time
        if self.in_reverse != in_reverse:
            self.direction_changed = current_time
        self.speed_index = speed_index
        self.in_reverse = in_reverse
        self.functions = functions
        self.last_update = current_time

    @property
    def target_speed(self):
        abs_target = self.model.key_speeds[self.speed_index]
        return -abs_target if self.in_reverse else abs_target

    @property
    def is_sound_on(self):
        return self.functions.get(self.model.sound_function, False)

    @property
    def current_reverse_time(self):
        return self.model.reverse_time_with_sound if self.is_sound_on else 0

    def integrate_to(self, t: float) -> Tuple[float, float, float]:
        """
        Computes the velocity and cumulative distance traveled at time `t > last_update`, assuming no new signal has been sent since.

        **The model**:

        * If the direction was changed in the last update, computes the time to full stop using `deceleration_on_reverse` and the idle time if sound is on.
        * Assumes the speed then increases linearly towards the target speed with `model.acceleration`.

        Args:
            t: Time.

        Returns:
            speed: Signed speed at time `t` in cm/s.
            sgn_distance: Total signed distance at time `t`.
            abs_distance: Total distance at time `t`.
        """
        speed = self.speed
        target_speed = self.target_speed
        # --- Idle while sound is blocking ---
        if self.is_sound_on:
            t0_if_startup = self.sound_switched_on + self.model.startup_time
            t0_if_reversed = self.direction_changed + self.model.reverse_time_with_sound  # ToDo add deceleration_on_reverse
            t0 = max(self.last_update, t0_if_startup, t0_if_reversed)
        else:
            t0 = self.last_update
        if t0 >= t:  # Still waiting for sound to finish
            return 0., self.sgn_distance, self.abs_distance
        # --- Accelerate + constant ---
        acc_duration = min(abs(target_speed - speed) / self.model.acceleration, t - t0)
        const_duration = t - t0 - acc_duration
        avg_speed_while_acc = (speed + target_speed) / 2
        sgn_distance = self.sgn_distance + avg_speed_while_acc * acc_duration + target_speed * const_duration
        abs_distance = self.abs_distance + abs(avg_speed_while_acc) * acc_duration + abs(target_speed) * const_duration
        return target_speed, sgn_distance, abs_distance


def kmh_to_cms(kmh):
    if isinstance(kmh, (tuple, list)):
        return tuple(kmh_to_cms(entry) for entry in kmh)
    return kmh / 3.6 * 100 / 87


def cms_to_kmh(cms):
    return cms / 100 * 3.6 * 87


def unmask(speeds):
    speeds = list(speeds)
    for i in range(1, len(speeds)):
        if speeds[i] is None:
            speeds[i] = speeds[i - 1]
    return speeds


def fit_speeds(measured: tuple):
    import scipy
    def loss(x):
        max_speed, exponent = x
        pred = speeds(max_speed, exponent)
        result = 0
        for m, p in zip(measured, pred):
            if measured is not None:
                result += (m - p) ** 2
        return result
    result = scipy.optimize.minimize(loss, (200., 1.3))
    print(result)
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pylab
    pylab.plot(measured)
    pylab.plot(speeds(*result.x))
    pylab.show()
    return result.x


if __name__ == '__main__':
    from fpme.train_def import SHUTTLE
    tracker = TrainTracker(model=SHUTTLE.model)
    tracker.set(speed_index=0, in_reverse=False, functions={2: True}, current_time=0.)
    tracker.set(speed_index=5, in_reverse=True, functions={2: True}, current_time=1.)
    # target speed = 40 cm/s, accel = 10 cm/s^2 -> reaches target at t=4
    print(tracker.integrate_to(2.))
    # top_speed, exponent = fit_speeds((0, 2, 5, 10, 15, 22, 30, 41, 51, 64, 77, 91, 106, 120, 136))
    # print(f"Top speed: {top_speed}, exponent: {exponent}")
