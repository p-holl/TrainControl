import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from fpme.speed_util import speeds


@dataclass(frozen=True)
class DrivingModel:
    key_speeds: Tuple[float] = speeds(60, 1.3)  # cm/s
    acceleration: float = 60.
    deceleration: float = 60.
    deceleration_on_reverse: float = 100.
    sound_function: Optional[int] = None  # function index. None if has no sound
    reverse_time_with_sound: float = 0.
    startup_time: float = 0.


@dataclass
class TrainTracker:
    model: DrivingModel
    # --- Signal state ---
    speed_index: int = 0  # current signal [0, 15]
    reverse: bool = False
    functions: Dict[int, bool] = field(default_factory=dict)
    last_update: float = -1  # perf_counter time
    # --- Model state ---
    speed: float = 0.  # Actual current speed in cm/s
    sgn_distance: float = 0.  # distance travelled in cm
    abs_distance: float = 0. # distance travelled in cm

    def set(self, speed_index: int, reverse: bool, functions: Dict[int, bool], current_time: float):
        direction_changed = self.reverse != reverse
        dt = current_time - self.last_update
        # --- Integrate until now ---
        current_speed, _ = self.get_speed_and_acceleration(current_time)
        # --- Update state ---
        self.speed_index = speed_index
        self.reverse = reverse
        self.functions = functions
        self.last_update = current_time

        v1, a1 = self.model.get_speed_and_acceleration(self.last_update)
        v2, a2 = self.model.get_speed_and_acceleration(current_time)
        t1 = self.model.last_update
        t2 = current_time
        dt = t2 - t1
        delta = ((v1+v2) / 2 - (a2-a1) / 12 * dt) * dt
        self.sgn_distance += delta
        if v2 * v1 >= 0:  # no velocity flip
            self.abs_distance += abs(delta)
        else:
            self.abs_distance += self.total_distance_with_velocity_flip(t1, t2, v1, v2, a1, a2)

    @property
    def is_sound_on(self):
        return self.functions.get(self.sound_function, False)

    @property
    def current_reverse_time(self):
        return self.reverse_time_with_sound if self.is_sound_on else 0

    def get_speed_and_acceleration(self, t: float) -> Tuple[float, float]:
        """Returns signed speed in cm/s at time `t >= t0` where `t0` is the last update time."""
        dt = t - self.last_update
        target_speed = self.key_speeds[self.speed_index] * (-1 if self.reverse else 1)
        if self.direction_changed:  # reverse signal sent at t0
            t_stopped = self.deceleration_on_reverse
            t_reversed = t_stopped + self.current_reverse_duration
            t_started = t_reversed + self.startup_time
        return self.speed + dt * self.acceleration, self.acceleration

    @staticmethod
    def total_distance_with_velocity_flip(t1, t2, v1, v2, a1, a2):
        # ToDo assume acceleration jumps from a1 to a2. Compute time of jump from v1, v2
        dt = t2 - t1
        jerk = (a2 - a1) / dt  # jerk = da/dt
        # --- Solve for tau (time since t1) when v(t) = 0      v(t) = v1 + a1 * tau + 0.5 * jerk * tau^2 = 0 ---
        A = 0.5 * jerk
        B = a1
        C = v1
        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            raise ValueError("Velocity does not reach zero between t1 and t2.")
        sqrt_disc = math.sqrt(discriminant)
        tau1 = (-B + sqrt_disc) / (2 * A) if A != 0 else -C / B
        tau2 = (-B - sqrt_disc) / (2 * A) if A != 0 else -C / B
        tau = None
        for root in [tau1, tau2]:  # Choose the root within (0, dt)
            if 0 < root < dt:
                tau = root
                break
        if tau is None:
            raise ValueError("No valid zero-crossing for velocity in the interval.")
        t_turn = t1 + tau
        a_turn = a1 + jerk * tau
        # --- Compute total distance ---
        d1 = 0.5 * abs(v1) * tau - (a_turn - a1) * tau ** 2 / 12  # Distance from t1 to t_turn
        tau2 = t2 - t_turn
        d2 = 0.5 * abs(v2) * tau2 - (a2 - a_turn) * tau2 ** 2 / 12  # Distance from t_turn to t2
        return d1 + d2


def kmh_to_cms(kmh):
    return kmh / 3.6 * 100 / 87


def cms_to_kmh(cms):
    return cms / 100 * 3.6 * 87
