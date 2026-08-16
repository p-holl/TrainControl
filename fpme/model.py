from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np


def speeds(s15, exponent=1.3):
    return (*np.linspace(0, s15 ** (1/exponent), 15) ** exponent,)


@dataclass(frozen=True)
class DrivingModel:
    key_speeds: Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float] = speeds(60, 1.3)  # cm/s
    acceleration: float = 60.
    deceleration: float = 60.
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
    direction_changed_at: Optional[float] = None  # time when direction was last changed
    in_reverse: bool = False
    sound_switched_on_at: Optional[float] = None  # time when sound was last switched on
    functions: Dict[int, bool] = field(default_factory=dict)
    # --- Pending event flags for current update ---
    _pending_direction_changed: bool = field(default=False, init=False, repr=False)
    _pending_sound_switched_on: bool = field(default=False, init=False, repr=False)
    # --- Model state ---
    speed: float = 0.  # Modeled signed speed (cm/s) (negative when reversed) at t=last_update
    sgn_distance: float = 0.  # distance traveled (cm) at t=last_update
    abs_distance: float = 0. # distance traveled (cm) at t=last_update

    def set(self, speed_index: int, in_reverse: bool, functions: Dict[int, bool], current_time: float):
        # --- Determine event flags BEFORE integrating (using old state) ---
        direction_changed = self.in_reverse != in_reverse
        sound_switched_on = functions.get(self.model.sound_function, False) and not self.is_sound_on
        
        # --- Store which events happen in THIS update (only set if event occurred) ---
        if direction_changed:
            self._pending_direction_changed = True
        else:
            self._pending_direction_changed = False
        
        if sound_switched_on:
            self._pending_sound_switched_on = True
        else:
            self._pending_sound_switched_on = False
        
        # --- Integrate to now ---
        if self.last_update == -1:  # initialization event
            self.speed, self.sgn_distance, self.abs_distance = 0., 0., 0.
        else:
            self.speed, self.sgn_distance, self.abs_distance = self.integrate_to(current_time)
        
        # --- Update state and timestamps ---
        self.speed_index = speed_index
        if direction_changed:
            self.direction_changed_at = current_time
        self.in_reverse = in_reverse
        if sound_switched_on:
            self.sound_switched_on_at = current_time
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

    @property
    def direction_changed(self) -> bool:
        """Returns True if direction changed in the most recent set call."""
        return self._pending_direction_changed

    @property
    def sound_switched_on(self) -> bool:
        """Returns True if sound switched on in the most recent set call."""
        return self._pending_sound_switched_on

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
        dt = t - self.last_update
        speed = self.speed
        sgn_distance = self.sgn_distance
        abs_distance = self.abs_distance
        target = self.target_speed

        # --- Handle direction change: decelerate to 0, then wait startup_time if sound is on ---
        # Check if direction changed in current set OR we're still within the recovery period from a previous direction change
        in_direction_recovery = (self.direction_changed_at is not None and 
                                self.direction_changed_at <= self.last_update and
                                t >= self.direction_changed_at)
        
        if (self.direction_changed or in_direction_recovery) and not self.sound_switched_on:
            # Time to decelerate current speed to zero using deceleration_on_reverse
            decel = self.model.deceleration_on_reverse
            t_stop = abs(speed) / decel if decel > 0 else 0.

            if dt <= t_stop:
                # Still decelerating toward zero
                direction = 1.0 if speed > 0 else (-1.0 if speed < 0 else 0.0)
                new_speed = speed - direction * decel * dt
                # Clamp so we don't overshoot past zero
                if direction > 0:
                    new_speed = max(new_speed, 0.0)
                elif direction < 0:
                    new_speed = min(new_speed, 0.0)
                avg_speed = (speed + new_speed) / 2.0
                sgn_distance += avg_speed * dt
                abs_distance += abs(avg_speed) * dt
                return new_speed, sgn_distance, abs_distance
            else:
                # Finish deceleration to zero
                avg_speed = speed / 2.0
                sgn_distance += avg_speed * t_stop
                abs_distance += abs(avg_speed) * t_stop
                speed = 0.0
                dt -= t_stop

                # Wait for reverse_time_with_sound (startup delay before moving in new direction)
                reverse_wait = self.current_reverse_time
                if dt <= reverse_wait:
                    # Still waiting, speed stays 0
                    return 0.0, sgn_distance, abs_distance
                else:
                    dt -= reverse_wait
                    # Fall through to acceleration phase below with speed=0

        # --- Handle sound-switch-on startup delay (only if not already consumed by reverse wait) ---
        if self.sound_switched_on and speed == 0.0 and target != 0.0:
            startup = self.model.startup_time
            if dt <= startup:
                return 0.0, sgn_distance, abs_distance
            else:
                dt -= startup

        # --- Accelerate/decelerate linearly toward target speed ---
        if speed < target:
            rate = self.model.acceleration
            t_reach = (target - speed) / rate if rate > 0 else 0.
        elif speed > target:
            rate = self.model.deceleration
            t_reach = (speed - target) / rate if rate > 0 else 0.
        else:
            rate = 0.
            t_reach = 0.

        if dt <= t_reach:
            new_speed = speed + (rate * dt if speed < target else -rate * dt)
            avg_speed = (speed + new_speed) / 2.0
            sgn_distance += avg_speed * dt
            abs_distance += abs(avg_speed) * dt
            return new_speed, sgn_distance, abs_distance
        else:
            # Reach target, then travel remaining time at constant target speed
            avg_speed = (speed + target) / 2.0
            sgn_distance += avg_speed * t_reach
            abs_distance += abs(avg_speed) * t_reach
            dt -= t_reach
            sgn_distance += target * dt
            abs_distance += abs(target) * dt
            return target, sgn_distance, abs_distance


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
    tracker.set(speed_index=6, in_reverse=True, functions={2: True}, current_time=1.0001)
    # target speed = 40 cm/s, accel = 10 cm/s^2 -> reaches target at t=4
    print(tracker.integrate_to(2.))
    # top_speed, exponent = fit_speeds((0, 2, 5, 10, 15, 22, 30, 41, 51, 64, 77, 91, 106, 120, 136))
    # print(f"Top speed: {top_speed}, exponent: {exponent}")
