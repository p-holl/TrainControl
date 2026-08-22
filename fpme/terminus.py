import json
import logging
import os.path
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from threading import Thread, Lock
from typing import Optional, List, Tuple, Dict

from fpme.audio import play_announcement, play_background_loop, async_play, set_background_volume
from fpme.helper import schedule_at_fixed_rate
from fpme.relay8 import Relay8, RelayManager
from fpme.train_control import TrainControl, TrainState, get_speed_index
from fpme.train_def import Train, ICE, S, E_RB, E_BW, E40, DAMPF, BEIGE, ROT, DIESEL, BUS, train_by_name, MW_TGV, SHUTTLE

# 6: first switch primary, 7: first switch secondary, 8: both secondary switches
SWITCH_STATE = {  # True -> open_channel, False -> close_channel
    1: {6: False, 8: True},
    2: {6: False, 8: False},
    3: {6: True, 7: True},
    4: {6: True, 7: False, 8: False},
    5: {6: True, 7: False, 8: True},
}
SECOND_SWITCH_DIST = {1: 36, 2: 36, 3: None, 4: 18, 5: 18}

PREVENT_EXIT = {  # when entering platform x, train on platforms y must wait
    1: [2, 3],
    2: [3],
    5: [4],
}

LIMIT_INDEX = {1: 0, 2: 1, 3: 0, 4: 0, 5: 1}  # speed limit index by platform

ENTRY_SIGNAL = 3
ENTRY_POWER = 4  # no power when open

CONTACT_OFFSET = -20  # distance (cm) how far the contact trigger extends beyond the board
KEEP_ENTRY_OPEN_SEC = 12


@dataclass
class ParkedTrain:
    train: Train
    state: TrainState
    prev_track: Optional[str]
    platform: int  # in {1, 2, 3, 4, 5}
    entry_speed_level: Optional[int]
    dist_request: float = None  # Signed distance when enter request was sent. None for trains set through the UI.
    dist_trip: float = None  # Signed distance when entering the switches
    time_trip: float = None
    dist_clear: float = None  # Signed distance when leaving the sensor, now fully on switches
    time_clear: float = None
    dist_reverse: float = None  # Abs distance when the 'reverse' button is clicked for the fist time
    # --- For departure sound ---
    time_stopped: float = None  # perf_counter() when train came to rest in station. Can be updated.
    dist_stopped: float = None  # Signed distance when last stopped.
    doors_closing: bool = False
    time_departed: float = None  # Track departure so we don't play sounds multiple times
    # --- For special announcements ---
    announcements_played = ()  # can contain 'connections', 'delay/real', 'delay/fake', 'general/real', 'general/fake'
    time_last_announcement = -100  # start time of last train announcement. Must be at least 15 seconds until next one.
    duration_last_announcement = 0  # 15s for delay reasons, 15? seconds for connections

    def __post_init__(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Creating ParkedTrain for {self.train}")

    @property
    def has_tripped_contact(self):
        return self.dist_trip is not None

    @property
    def has_cleared_contact(self):
        return self.dist_clear is not None

    def mark_cleared_contact(self):
        self.dist_clear = self.state.signed_distance
        self.time_clear = time.perf_counter()

    @property
    def train_length(self):
        computed_length = abs(self.dist_clear - self.dist_trip) - 0.18  # detector track length
        return min(120., computed_length)

    @property
    def entered_forward(self):
        if self.dist_clear is None:
            return (self.dist_trip - self.dist_request) > 0
        else:
            return (self.dist_clear - self.dist_trip) > 0

    @property
    def was_entry_recorded(self):
        return self.dist_request is not None

    @property
    def has_reversed(self):
        return self.dist_reverse is not None

    def get_position(self):
        """Positive towards station."""
        if not self.has_tripped_contact:
            return None
        if not self.was_entry_recorded:  # added by UI
            return 220 - (self.state.abs_distance - self.dist_reverse)  # from middle of platform
        delta = self.state.signed_distance - self.dist_trip
        if not self.entered_forward:  # entered_forward only available if was_entry_recorded
            delta = -delta  # make sure positive in station
        if self.has_reversed:  # here it's hard to know which direction the train is going.
            since_rev = self.state.abs_distance - self.dist_reverse
            return min(300., delta) - since_rev * .8  # safety margin
        return CONTACT_OFFSET + delta

    def get_end_position(self):
        return self.get_position() - self.train_length

    @cached_property
    def delay_minutes(self):
        if random.random() < self.train.info.delay_rate:
            return random.randint(1, self.train.info.max_delay)
        else:
            return 0

    def __repr__(self):
        status = 'cleared' if self.has_cleared_contact else ('tripped' if self.has_tripped_contact else 'requested')
        return f"{self.train} on platform {self.platform} ({status})."


# @dataclass
# class TrackedTrain:
#     train: Train
#     state: TrainState
#     track: str
#     dist_start: float
#     num_reverse: int
#
#     def


class Terminus:

    def __init__(self, relay: Relay8, control: TrainControl, port: str, measure=False):
        # Setup logging with timestamped filename
        log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"terminus-log-{log_timestamp}.txt"
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info("=" * 80)
        self.logger.info(f"Terminus initialization started - Log file: {log_filename}")
        self.logger.info("=" * 80)
        
        assert control.generator.is_open(port), f"Terminus cannot be managed without entry sensor but {port} is not open."
        self.logger.debug(f"Entry sensor port verified: {port}")
        
        self.relay = relay
        self.control = control
        self.port = port
        self.measure = measure
        self.trains: List[ParkedTrain] = []  # trains in Terminal
        self.entering: Optional[ParkedTrain] = None
        self.correcting = {}  # Train -> direction
        self._request_lock = Lock()
        
        self.logger.debug("Initializing relay channels...")
        relay.close_channel(1)
        relay.close_channel(2)
        relay.close_channel(ENTRY_SIGNAL)
        relay.open_channel(ENTRY_POWER)
        self.logger.debug("Relay channels initialized")
        
        self.logger.info("Loading saved terminus state...")
        self.load_state()
        
        for t in self.trains:
            t.state.set_speed_limit('terminus', t.train.info.max_speed_in_station[LIMIT_INDEX[t.platform]], new_track='terminus')
        
        self.logger.info(f"Scheduling periodic tasks...")
        schedule_at_fixed_rate(self.save_state, 5.)
        schedule_at_fixed_rate(self.check_exited, 1.)
        schedule_at_fixed_rate(self.update, 0.1)
        
        self.logger.debug("Starting background ambient loop...")
        play_background_loop(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'sound', 'ambient', 'station.mp3')))
        
        self.logger.info(f"Terminus initialization complete. Loaded {len(self.trains)} trains from saved state. Measurement mode: {measure}")

    def save_state(self, *_args):
        self.logger.debug("Saving terminus state to terminus.json")
        data = {
            'switches': [],
            'trains': [{
                'name': t.train.id,
                'platform': t.platform,
                'sgn_dist': t.state.signed_distance,
                'abs_dist': t.state.abs_distance,
                'dist_request': t.dist_request,
                'dist_trip': t.dist_trip,
                'dist_clear': t.dist_clear,
                'dist_reverse': t.dist_reverse,
            } for t in self.trains]
        }
        with open("terminus.json", 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        self.logger.debug(f"State saved: {len(self.trains)} trains")

    def load_state(self):
        if not os.path.isfile("terminus.json"):
            self.logger.info("No saved terminus state found (terminus.json does not exist)")
            return
        self.logger.info("Loading saved terminus state from terminus.json")
        with open("terminus.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        for train_data in data['trains']:
            train = train_by_name(train_data['name'])
            platform = train_data['platform']
            dist_request = train_data['dist_request']
            dist_trip = train_data['dist_trip']
            dist_clear = train_data['dist_clear']
            dist_reverse = train_data['dist_reverse']
            state = self.control[train] if train in self.control else TrainState(self.control, train, set(), {})
            sgn_delta = state.signed_distance - train_data['sgn_dist']
            abs_delta = state.abs_distance - train_data['abs_dist']  # typically < 0
            self.trains.append(ParkedTrain(train, state, 'terminus', platform, None,
                                           dist_request=dist_request + sgn_delta if dist_request is not None else None,
                                           dist_trip=dist_trip + sgn_delta if dist_trip is not None else None,
                                           time_trip=-100,
                                           dist_clear=dist_clear + sgn_delta if dist_clear is not None else None,
                                           dist_reverse=dist_reverse + abs_delta if dist_reverse is not None else state.abs_distance,  # train could be reversed by restart
                                           doors_closing=False,
                                           dist_stopped=state.signed_distance,
                                           time_stopped=-100,))
            self.logger.info(f"Restored train from saved state: {train} on platform {platform}")
            if train in READY_SOUNDS:
                state.custom_acceleration_handler = self.handle_acceleration
                self.logger.debug(f"Terminus blocking acceleration on {train} - handler id: {id(state)}")
            else:
                self.logger.warning(f"No sound available for train: {train}")

    def reverse_to_exit(self):
        for train in self.trains:
            if train.dist_trip is not None and train.state.abs_distance == 0 and train.entered_forward != train.state.is_in_reverse:
                self.control.reverse(train.train, 'terminus', auto_activate=False)

    def get_train_position(self, train: Train):
        for t in self.trains:
            if t.train == train:
                return t.platform, t.get_position()
        return None, None

    def set_occupied(self, platform: int, train: Train):
        self.logger.debug(f"Setting platform {platform} as occupied for train {train}")
        state = self.control[train] if train in self.control else TrainState(self.control, train, set(), {}, {})
        state.track = 'terminus'
        if self.entering is not None and self.entering.train == train:
            self.clear_entering()
        if any([t.train == train for t in self.trains]):
            t = [t for t in self.trains if t.train == train][0]
            t.platform = platform
            self.logger.info(f"Moved {train} to platform {platform}")
        else:
            dist = state.signed_distance
            abs_dist = state.abs_distance
            position = 300
            train_length = 50
            t = ParkedTrain(train, state, state.track, platform, None, None, dist_trip=dist - position - CONTACT_OFFSET, dist_clear=dist - position - train_length - 0.36 - CONTACT_OFFSET, dist_reverse=abs_dist, time_stopped=-100)
            self.trains.append(t)
            self.logger.info(f"Added new train via UI: {t}")
        self.logger.debug(f"Current trains in terminus: {self.trains}")

    def set_empty(self, platform: int):
        self.logger.info(f"Clearing platform {platform}")
        trains = [t for t in self.trains if t.platform == platform]
        for t in trains:
            t.state.track = 'regional' if t.platform <= 3 else 'high-speed'
            t.state.set_speed_limit('terminus', None, cause='terminus')
            self.logger.debug(f"Released speed limit for {t.train}, track set to {t.state.track}")
        self.trains = [t for t in self.trains if t.platform != platform]
        if self.entering is not None and self.entering.platform == platform:
            self.entering.state.track = self.entering.prev_track
            self.clear_entering()
        self.logger.debug(f"Current trains in terminus: {self.trains}")

    def is_in_terminus(self, train: Train):
        return any(t.train == train for t in self.trains)

    def remove_train(self, train: Train):
        if self.entering and self.entering.train == train:
            self.entering.state.track = self.entering.prev_track
            self.clear_entering()
            self.logger.info(f"Removed entering train {train}")
        else:
            trains = [t for t in self.trains if t.train == train]
            if trains:
                for t in trains:
                    t.state.track = t.prev_track
                    self.control.set_speed_limit(t.train, 'terminus', None)
                self.trains = [t for t in self.trains if t.train != train]
                self.logger.info(f"Removed train {train} from terminus")
            else:
                self.logger.warning(f"Could not remove {train} from terminus - train not found")

    def correct_move(self, train: Train, direction: float):
        self.correcting[train] = direction

    def on_reversed(self, train: Train):
        for t in self.trains:
            if t.train == train:
                if t.dist_reverse is not None:
                    self.logger.debug(f"Ignoring subsequent reverse event for {train}")
                    continue  # ignore subsequent reverses
                t.dist_reverse = t.state.abs_distance
                self.logger.info(f"Train reversed: {train} at position {t.dist_reverse:.2f}cm")
                if self.measure and t.entry_speed_level is not None:
                    time_since_clear = time.perf_counter() - t.time_clear
                    time_since_trip = time.perf_counter() - t.time_trip
                    self.logger.debug(f"Measurement recorded: entry_speed_level={t.entry_speed_level}, trip_time={time_since_trip:.2f}s, clear_time={time_since_clear:.2f}s")
                    write_measurement(train.id, t.platform, t.entry_speed_level, time_since_trip, time_since_clear)
                if t.train in READY_SOUNDS:
                    self.logger.debug(f"Setting custom acceleration handler for {t} (sound: {READY_SOUNDS[t.train][0]})")
                    t.state.custom_acceleration_handler = self.handle_acceleration

    def handle_acceleration(self, train: Train, controller: str, acc_input: float, cause: str):
        for t in self.trains:
            if t.train == train:
                if acc_input > 0:
                    if t.doors_closing:
                        self.logger.debug(f"Ignoring acceleration input for {train} - doors already closing")
                        return
                    t.doors_closing = True
                    self.logger.debug(f"Doors closing initiated for {train} on platform {t.platform}")
                    if self.control.sound < 2:
                        t.state.custom_acceleration_handler = None
                        self.logger.debug(f"Sound level {self.control.sound} - skipping door sound")
                        break
                    # --- Play sound ---
                    sound, duration, vol = READY_SOUNDS[t.train]
                    duration -= t.train.model.reverse_time_with_sound
                    self.logger.debug(f"Playing door closing sound: {sound} (blocking for: {duration:.1f}s)")
                    async_play('departure-effects/' + sound, int(t.platform <= 3) * vol, int(t.platform > 3) * vol)
                    # --- Wait, then release control ---
                    def release_block(t=t):
                        duration > 0 and time.sleep(duration)
                        self.logger.info(f"Door closing complete for {t.train} on platform {t.platform}")
                        t.state.custom_acceleration_handler = None
                    Thread(target=release_block).start()
                break
        else:
            self.logger.warning(f"Terminus received acceleration input for {train} but train is not in station")

    def request_entry(self, train: Train):  # Button C
        self.logger.info(f"{train} requests entry. Currently in terminus: {[t.train.name for t in self.trains]}")
        is_in_station = any(t.train == train for t in self.trains)
        with self._request_lock:
            self.logger.debug(f"Entry lock acquired. Currently entering train: {self.entering.train if self.entering else 'None'}")
            if not is_in_station and self.entering:
                if train == self.entering.train:  # clicked again, no effect
                    self.logger.debug(f"{train} entry request ignored - already being processed")
                    return
                elif self.entering.has_tripped_contact:
                    if not self.entering.has_cleared_contact:
                        self.logger.warning(f"{train} cannot enter until {self.entering} has cleared contact")
                        self.control.force_stop(train, "wait for previous train")  # Wait until previous train has passed
                        return
                else:  # Who is first? Previous one might have been an accident. Stop both, block entry
                    self.logger.error(f"Entry conflict between {train} and {self.entering.train} - emergency stop both")
                    self.control.emergency_stop(train, f"Contested terminus entry: {train} vs {self.entering.train}")
                    self.control.emergency_stop(self.entering.train, f"Contested terminus entry: {train} vs {self.entering.train}")
                    self.clear_entering()
                    return
            if is_in_station:
                t = [t for t in self.trains if t.train == train][0]
                self.logger.info(f"{train} is already in terminus at platform {t.platform}, position: {t.get_position():.1f}cm, cleared={t.has_cleared_contact}")
                # --- Play sound if parked ---
                if t.state.speed == 0 and self.control.sound >= 1 and len(t.announcements_played) < 2 and time.perf_counter() > t.time_last_announcement + t.duration_last_announcement and t.time_stopped is not None:
                    self.logger.debug(f"Playing connections announcement for {train}")
                    self.play_connections(t)
                else:
                    self.logger.debug(f"Cannot play announcement - sound={self.control.sound}, speed={t.state.speed}, announcements_played={len(t.announcements_played)}")
                return
            # --- prepare entry ---
            platform = select_track(train, self.get_platform_state())
            self.logger.info(f"{train} assigned to platform {platform}")
            if platform is None:  # cannot enter
                self.logger.warning(f"{train} cannot enter - no available platform, forcing stop")
                self.control.force_stop(train, "no platform")
                return
            state = self.control[train]
            self.entering = entering = ParkedTrain(train, state, state.track, platform, get_speed_index(state, 0., True))
            entering.dist_request = entering.state.signed_distance
            entering.state.restore_speed_after_reset = True
            self.trains.append(entering)
        self.control.set_speed_limit(train, 'terminus', train.info.max_speed_in_station[LIMIT_INDEX[platform]])
        self.prevent_exit(platform)
        self.relay.open_channel(ENTRY_SIGNAL)
        self.relay.close_channel(ENTRY_POWER)
        self.logger.debug(f"Entry signal opened, entry power closed for {train}")
        if self.control.sound >= 1:
            self.logger.debug(f"Playing entry announcement for {train} to platform {platform}")
            play_entry_announcement(train, platform, entering.delay_minutes)

        def process_entry(entering: ParkedTrain, duration=KEEP_ENTRY_OPEN_SEC, interval=0.01, max_train_length=130):
            for _ in range(int(duration / interval)):
                if not entering.train.trips_contacts and abs(entering.state.signed_distance - entering.dist_request) >= 20:
                    set_switches_for(self.relay, entering.platform, 90., -10, detection_time=5.)
                    break
                if self.control.generator.contact_status(self.port)[0]:
                    self.logger.info(f"Entry contact tripped for {entering}")
                    break
                time.sleep(interval)
            else:  # --- not tripped - maybe button pressed on accident or train too far ---
                self.logger.warning(f"Entry timeout: {entering.train} did not trip entry contact within {KEEP_ENTRY_OPEN_SEC}s, aborting entry")
                entering.state.set_speed_limit('terminus', None, new_track=entering.prev_track)
                self.clear_entering()
                self.control.emergency_stop(train, "train did not enter terminus")
                if entering in self.trains:
                    self.trains.remove(entering)
                return
            # --- Contact tripped ---
            self.logger.info("Entry contact tripped, processing train entry")
            entering.dist_trip = entering.state.signed_distance
            entering.time_trip = time.perf_counter()
            entering.entry_speed_level = get_speed_index(entering.state, 0., True)  # update recorded speed for measurement
            entering.state.track = 'terminus'
            self.logger.debug(f"Train entered at distance {entering.dist_trip:.2f}cm, speed_level={entering.entry_speed_level}")
            if entering.dist_trip == entering.dist_request:
                entering.dist_request -= -1e-3 if entering.state.is_in_reverse else 1e-3
            driven = entering.dist_trip - entering.dist_request
            if (entering.state.speed > 0) != entering.entered_forward:
                self.logger.warning(f"Direction mismatch during entry: driven={driven:.2f}cm, speed={entering.state.speed}, forward={entering.entered_forward}")
            # --- async switches and signal ---
            if entering.train.trips_contacts:
                self.logger.debug(f"Setting switches for {entering.train} on platform {entering.platform}")
                set_switches_for(self.relay, platform, entering.train.info.max_speed_in_station[LIMIT_INDEX[entering.platform]], CONTACT_OFFSET)
            def red_when_entered():
                while True:
                    time.sleep(0.1)
                    if entering.get_position() > 20:
                        self.relay.close_channel(ENTRY_SIGNAL)  # red when train has driven for 20cm
                        self.logger.debug(f"Entry signal closed (red) - train {entering.train} has progressed 20cm")
                        return
            self.logger.debug("Starting async signal closure thread")
            Thread(target=red_when_entered).start()
            # --- wait for clear sensor ---
            self.logger.debug("Waiting for entry contact to clear")
            while True:
                time.sleep(interval)
                # print(f"Sensor: {self.control.generator.contact_status(self.port)[0]}")
                # --- External trains ---
                if not train.trips_contacts:
                    if entering.dist_clear is None and entering.get_position() > CONTACT_OFFSET + max_train_length:
                        entering.mark_cleared_contact()
                # --- Managed trains ---
                elif not self.control.generator.contact_status(self.port)[0]:  # possible sensor clear
                    if entering.dist_clear is None:
                        self.logger.debug("Entry sensor cleared, waiting for possible re-trigger")
                        entering.mark_cleared_contact()
                elif entering.dist_clear is not None and entering.get_end_position() < CONTACT_OFFSET + 30:  # another wheel entered
                    self.logger.debug("Another wheel detected entering contact")
                    entering.dist_clear = None  # enable above block to re-trigger
                    continue
                if entering.dist_clear is None and entering.get_position() > CONTACT_OFFSET + max_train_length:
                    entering.mark_cleared_contact()
                    self.logger.debug(f"Max train length reached. End position: {entering.get_end_position():.2f}cm")
                # --- cleared switches ---
                if self.entering is not None and self.entering.dist_clear is not None and entering.get_end_position() > 60:  # approx. 57 cm
                    self.logger.info(f"Entry complete: {entering.train} cleared switches on platform {entering.platform}")
                    self.clear_entering()
                    return

        Thread(target=process_entry, args=(entering,)).start()

    def check_exited(self, *_):
        for t in tuple(self.trains):
            if t.has_cleared_contact:
                pos = t.get_position()
                exited = pos < 0
                if exited:
                    self.trains.remove(t)
                    t.state.set_speed_limit('terminus', None, new_track='regional' if t.platform <= 3 else 'high-speed')
                    self.logger.info(f"{t.train} exited the terminus from platform {t.platform}")

    def play_connections(self, t: ParkedTrain):
        logger = logging.getLogger(__name__)
        logger.debug(f"Checking for announcements: {t.announcements_played}")
        if 'connections' not in t.announcements_played and time.perf_counter() < t.time_stopped + 15:  # first announcement is about other trains in station (only if any)
            passenger_trains = [t_ for t_ in self.trains if t_ != t and t_.train.is_passenger_train and (t_.state.speed == 0 or (t_.state.speed > 0) == t_.entered_forward)]
            if passenger_trains:
                connections = [(t_.train, t_.platform) for t_ in passenger_trains]
                t.announcements_played += ('connections',)
                t.time_last_announcement = time.perf_counter()
                t.duration_last_announcement = play_connections(t.platform, connections)
                logger.info(f"Playing connection announcement for {t.train} on platform {t.platform}: {len(connections)} other trains")
        else:
            t.announcements_played += ('delay',)
            t.time_last_announcement = time.perf_counter()
            t.duration_last_announcement = play_special_announcement(t.train, t.platform, t.delay_minutes, time.perf_counter() - t.time_stopped)
            logger.info(f"Playing delay announcement for {t.train} on platform {t.platform}: {t.delay_minutes} minutes delay")

    def update(self, *_):
        set_background_volume(.2 if self.control.sound >= 2 else 0)
        for train in self.trains:
            if not train.state.speed and train.has_cleared_contact:  # stopped after contact
                if train.time_stopped is None:
                    self.logger.info(f"{train.train} came to a stop on platform {train.platform}")
                    train.time_stopped = time.perf_counter()
                    train.dist_stopped = train.state.signed_distance
                    if self.entering == train:
                        self.clear_entering()
                    def delayed_play(train=train):
                        time.sleep(2.)
                        self.play_connections(train)
                    Thread(target=delayed_play).start()
                elif not train.has_reversed and train.state.signed_distance != train.dist_stopped:  # Continued a bit further and stopped again
                    self.logger.debug(f"{train.train} stopped again on platform {train.platform}, moved {abs(train.state.signed_distance - train.dist_stopped):.2f}cm from previous position")
                    train.time_stopped = time.perf_counter()
                    train.dist_stopped = train.state.signed_distance
            elif train.time_departed is None and train.time_stopped is not None and train.has_reversed and train.state.speed:
                self.logger.info(f"{train.train} is departing from platform {train.platform}")
                train.time_departed = time.perf_counter()
                if self.control.sound >= 2 and self.control.is_power_on(train.train) and train.train in DEPARTURE_SOUNDS:
                    if time.perf_counter() - train.time_stopped > 4.:
                        sound, vol = DEPARTURE_SOUNDS[train.train]
                        self.logger.debug(f"Playing departure sound: {sound}")
                        async_play("departure/"+sound, int(train.platform <= 3) * vol, int(train.platform > 3) * vol)
            # --- Manual position correction ---
            if train.train in self.correcting and not train.state.speed:
                direction = self.correcting[train.train] * (-2 if train.entered_forward else 2)
                if direction:
                    if train.dist_trip is not None:
                        train.dist_trip -= direction
                    if train.dist_reverse is not None:
                        train.dist_reverse += direction
                    self.logger.debug(f"Position corrected for {train.train}: {direction}cm")
        if self.entering is not None and self.entering.time_trip and time.perf_counter() - self.entering.time_trip > 20:
            self.logger.warning(f"{self.entering.train} has been entering for {time.perf_counter() - self.entering.time_trip:.1f}s, clearing entry")
            self.clear_entering()

    def prevent_exit(self, entering_platform):
        if entering_platform == 1:
            self.relay.close_channel(1)  # Platforms 2, 3
            self.logger.debug(f"Blocked exit from platforms 2,3 to allow platform 1 entry")
        elif entering_platform == 2:
            self.relay.close_channel(1)  # Platforms 2, 3
            self.logger.debug(f"Blocked exit from platforms 2,3 to allow platform 2 entry")
        elif entering_platform == 5:
            self.relay.close_channel(2)  # Platform 4
            self.logger.debug(f"Blocked exit from platform 4 to allow platform 5 entry")
        trains = [t for t in self.trains if t.platform in PREVENT_EXIT.get(entering_platform, [])]
        for t in trains:
            if (t.state.speed < 0) == t.entered_forward:
                self.control.emergency_stop(t.train, 'terminus-conflict')
                t.state.set_speed_limit('terminus-wait', 0)

    def clear_entering(self):
        if self.entering is not None:
            self.entering.state.restore_speed_after_reset = False
            self.logger.debug(f"Clearing entry process for {self.entering.train}")
        self.entering = None
        self.relay.close_channel(ENTRY_SIGNAL)
        self.relay.open_channel(ENTRY_POWER)
        self.free_exit()

    def free_exit(self):
        self.logger.debug("Freeing exit paths for all platforms")
        self.relay.open_channel(1)  # Platforms 2, 3
        self.relay.open_channel(2)  # Platform 4
        for t in self.trains:
            t.state.set_speed_limit('terminus-wait', None)

    def get_platform_state(self):
        """For each platform returns one of (empty, parked, entering, exiting) """
        state = {i: 'empty' for i in range(1, 6)}
        for t in self.trains:
            speed = t.state.speed
            if speed == 0:
                state[t.platform] = 'parked'
            elif (speed > 0) == t.entered_forward:
                state[t.platform] = 'entering'
            else:
                state[t.platform] = 'exiting'
        return state

    def reset_switches(self):
        for channel in range(6, 9):
            self.relay.open_channel(channel)
            time.sleep(.2)
        for channel in range(6, 9):
            self.relay.close_channel(channel)
            time.sleep(.2)


def select_track(train: Train, state: Dict[int, str]):
    """ Returns `None` if the train cannot enter because of collisions. """
    logger = logging.getLogger(__name__)
    can_enter = {
        1: state[1] == 'empty' and state[2] != 'exiting' and state[3] != 'exiting',
        2: state[2] == 'empty' and state[3] != 'exiting',
        3: state[3] == 'empty',
        4: state[4] == 'empty',
        5: state[5] == 'empty' and state[4] != 'exiting',
    }
    can_enter = [p for p, c in can_enter.items() if c]
    if not can_enter:
        logger.warning(f"{train.name} cannot enter - no available platforms. Platform states: {state}")
        return None
    regional = random.random() < train.info.regional_prob
    cost_regional = int(not regional)
    cost_far_distance = int(regional)
    sw_av = .25 * train.info.switch_avoidance
    base_cost = {
        1: cost_regional + 0. + sw_av,
        2: cost_regional + .1,
        3: cost_regional + .25 - sw_av,
        4: cost_far_distance + -.1 + sw_av,
        5: cost_far_distance,
    }
    cost = {p: base_cost[p] for p in can_enter}
    best = min(cost, key=cost.get)
    logger.info(f"{train.name} assigned to platform {best} (costs={cost})")
    return best


def set_switches_for(relay, platform: int, train_speed, train_position, speed_margin=0.5, detection_time=1.0):
    logger = logging.getLogger(__name__)
    train_speed_cm_s = train_speed / 87 / 3.6 * 100 + speed_margin
    time_to_1 = (16 - train_position) / train_speed_cm_s - detection_time
    logger.debug(f"Planning switches for platform {platform} - train speed: {train_speed_cm_s:.1f} cm/s, ETA to first switch: {time_to_1:.1f}s")
    target = SWITCH_STATE[platform]
    def async_set_switches():
        if 7 in target:
            time_to_1 > 0 and time.sleep(time_to_1 - .1)
            relay.set_channel_open(6, target[6])
            time.sleep(.1)
            relay.set_channel_open(7, target[7])
        else:
            time_to_1 > 0 and time.sleep(time_to_1)
            relay.set_channel_open(6, target[6])
        if 8 in target:
            dt = SECOND_SWITCH_DIST[platform] / train_speed_cm_s
            time.sleep(dt)
            relay.set_channel_open(8, target[8])
    Thread(target=async_set_switches).start()


TARGETS = {
    ICE: {
        1: ('I C E, 86',  "Waldbrunn, über: Neuffen"),
        2: ('I C E, 29', "Neuffen"),
        3: ('I C E, 52', "Wiesbaden, über: Waldbrunn"),
        4: ('I C E, 18',  "Böblingen"),
        5: ('I C E, 34',  "Radeburg, über: Waldbrunn"),
    },
    S: {
        1: ('S 3', "Waldbrunn"),
        2: ('S 5', "Neuffen"),
        3: ('S 1', "Kirchbach"),
        4: ('S zwo', "Böblingen"),
        5: ('S 4', "Kleiningen"),
    },
    BUS: {
        1: ('Schienenbus', "Waldbrunn"),
        2: ('Schienenbus', "Neuffen"),
        3: ('Schienenbus', "Kirchbach"),
        4: ('Schienenbus', "Böblingen"),
        5: ('Schienenbus', "Kleiningen"),
    },
    E_BW: {
        1: ("Intercity", "Waldbrunn, über: Neuffen"),
        2: ("Intercity", "Neuffen"),
        3: ("Intercity", "Wiesbaden, über: Waldbrunn"),
        4: ("Intercity", "Böblingen"),
        5: ("Intercity", "Radeburg, über: Waldbrunn"),
    },
    E_RB: {
        1: ("Regionalbahn", "Waldbrunn"),
        2: ("Regionalbahn", "Neuffen"),
        3: ("Regionalbahn", "Kirchbach"),
        4: ("Regionalbahn", "Böblingen"),
        5: ("Regionalbahn", "Kleiningen"),
    },
    ROT: {
        1: ("Regional-Express", "Waldbrunn, über: Neuffen"),
        2: ("Regional-Express", "Neuffen"),
        3: ("Regional-Express", "Wiesbaden"),
        4: ("Regional-Express", "Böblingen"),
        5: ("Regional-Express", "Radeburg"),
    },
    BEIGE: {
        1: ("Nahverkehrszug", "Waldbrunn"),
        2: ("Nahverkehrszug", "Neuffen"),
        3: ("Nahverkehrszug", "Wiesbaden"),
        4: ("Eilzug", "Böblingen"),
        5: ("Eilzug", "Radeburg"),
    },
    MW_TGV: {
        1: ('Tee - Schee - Weh',  "Paris, über: Neuffen"),
        2: ('Tee - Schee - Weh', "Straßburg"),
        3: ('Tee - Schee - Weh', "Marseille, über: Wiesbaden"),
        4: ('Tee - Schee - Weh',  "Lyon, über: Böblingen"),
        5: ('Tee - Schee - Weh',  "Toulouse, über: Radeburg"),
    },
    SHUTTLE: {
        1: ("Regional-Express", "Waldbrunn, über: Neuffen"),
        2: ("Regional-Express", "Neuffen"),
        3: ("Regional-Express", "Wiesbaden"),
        4: ("Regional-Express", "Böblingen"),
        5: ("Regional-Express", "Radeburg"),
    },
}


def play_entry_announcement(train: Train, platform: int, delay_minutes: int):
    logger = logging.getLogger(__name__)
    if train in TARGETS:
        name, target = TARGETS[train][platform]
        hour, minute, delay = delayed_now(delay_minutes)
        delay_text = f", heute circa {delay} Minuten später." if delay else ". Vorsicht bei der Einfahrt."
        speech = f"Gleis {PL_NUM[platform]}, Einfahrt. {name}, nach: {target}, Abfahrt {hour} Uhr {minute}{delay_text}"
    else:
        speech = f"Vorsicht auf Gleis {platform}, ein Zug fährt ein."
    logger.info(f"Playing entry announcement: '{speech}'")
    play_announcement(speech, left_vol=int(platform <= 3), right_vol=int(platform > 3))


OPPOSITE = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: None,
}
PL_NUM = {
    1: "eins",
    2: "zwo",
    3: "drei",
    4: "vier",
    5: "fünf"
}


def play_connections(platform: int, connections: List[Tuple[Train, int]]):
    if len(connections) > 2:
        connections = random.sample(connections, 2)
    texts = []
    for train, pl in connections:
        if train in TARGETS:
            name, target = TARGETS[train][pl]
            texts.append(f"{name}, nach: {target} von Gleis {PL_NUM[pl]}{', direkt gegenüber' if OPPOSITE[platform] == pl else ''}.")
    play_announcement(' '.join(texts), left_vol=int(platform <= 3), right_vol=int(platform > 3), cue='anschlüsse', cue_vol=1.)
    return 2 + 4.5 * len(texts)


def play_special_announcement(train: Train, platform: int, delay_minutes: int, entered_seconds_ago: float):
    sentences = [
        "Information zu, Hoggworts Express, nach: Hoggworts: Heube ab Gleis 8 Drei Viertel, direkt gegenüber.",
        "I C E 397, nach: Atlantis, fällt heute aus.",
        "Achtung Passagiere des Polarexpresses: Bitte halten Sie Ihr goldenes Ticket bereit.",
        "Information zu Orient Express: Der Zug verspätet sich aufgrund eines Mordes an Bord.",
        "Information zu: Thomas der kleinen Lokomotive: Heute ca. 20 Minuten später, da sie einem Freund auf die Gleise hilft.",
        "Information zu: Zeitreisezug, nach: 1955. Bitte vermeiden Sie Paradoxa",
        "Information zum Schienenersatzverkehr zwischen: München, und: Berlin. Bitte benutzen Sie eines der bereitstehenden Fahrräder.",
        "Information zu: I C E, 86, Heute pünktlich. Grund hierfür sind Personen im Gleis, die den Zug anschieben.",
        "Achtung Passagiere des I C E 987, nach: Gotham Sittie. Bitte benutzen Sie ausschließlich Abschnitte D bis F., Grund hierfür ist ein Auftritt des Jokers in Abteil A.",
        "Achtung Passagiere des I C E 456, nach: Wunderland. Bitte folgen Sie dem weißen Kaninchen zum Gleis",
        "Bitte lassen Sie Ihr Gepäck nicht unbeaufsichtigt. Sollte Ihnen alleinstehendes Gepäck auffallen, tragen Sie es bitte aus dem Bahnhof.",
        "Letzter Aufruf für Passagier Hubert Bauer, gebucht auf ICE 410 nach Köln. Bitte begeben Sie sich umgehend zum Bahnsteig 3.",
        # "Jim Knopf", / SEV / Tauben / Bordrestaurant teuer
    ]
    real_reasons = [
        "sind Gegenstände im Gleis.",
        "ist eine Störung im Betriebsablauf.",
        "sind Verzögerungen im Betriebsablauf.",
        "sind polizeiliche Ermittlungen.",
        "ist ein Notarzteinsatz im Zug.",
        "ist ein Notarzteinsatz auf der Strecke.",
        "ist eine technische Störung an der Strecke.",
        "ist eine technische Störung am Zug.",
        "ist eine Signalstörung.",
        "sind Personen im Gleis.",
        "ist ein Unfall mit Personenschaden.",
        "ist die ärztliche Versorgung eines Fahrgastes.",
        "ist eine behördliche Maßnahme.",
        "ist eine defekte Tür.",
        "ist die Bereitstellung weiterer Wagen.",
        "ist ein defektes Stellwerk.",
        "ist eine Ober-Leitungs-Störung.",
        "ist ein Polizeieinsatz.",
        "ist eine Reparatur am Zug.",
        "ist eine Reparatur an der Oberleitung.",
        "ist eine Reparatur an einem Signal.",
        "ist die Reparatur an einer Weiche.",
        "ist eine Reparatur an der Strecke.",
        "ist eine Streckensperrung.",
        "ist eine Weichenstörung.",
        "sind Streikauswirkungen.",
        "ist ein technischer Defekt an einem anderen Zug.",
        "sind Tiere auf der Strecke.",
        "sind unbefugte Personen auf der Strecke.",
        "ist die Unterstützung beim Ein- und Ausstieg.",
        "ist ein Unwetter.",
        "ist die verspätete Bereitstellung des Zuges.",
        "ist eine Verspätung aus vorheriger Fahrt.",
        "ist die Verspätung eines vorausfahrenden Zuges.",
        "ist eine Verspätung im Ausland.",
        "ist die Vorfahrt eines anderen Zuges.",
        "ist das Warten auf Anschlussreisende.",
        "ist das Warten auf einen entgegenkommenden Zug.",
        "sind witterungsbedingte Beeinträchtigungen.",
        "sind Tiere im Gleis.",
        "sind ausgebrochene Tiere im Gleis.",
        "sind Bauarbeiten.",
        "ist eine behobene Störung am Zug.",
        "ist eine behobene Störung am Gleis.",
        "ist ein zusätzlicher Halt zum Ein- und Ausstieg.",
        "ist eine derzeit eingeschränkte Verfügbarkeit der Gleise.",
        "ist die Entschärfung einer Fliegerbombe.",
        "ist ein Feuerwehreinsatz auf der Strecke.",
        "ist ein kurzfristiger Personalausfall.",
        "ist eine Pass-und Zollkontrolle.",
        "sind Streikauswirkungen.",
        "ist eine technische Untersuchung am Zug.",
        "ist ein technischer Defekt an einem anderen Zug.",
        "ist die Umleitung des Zuges.",
        "ist ein umgestürzter Baum auf der Strecke.",
        "ist ein Unfall an einem Bahnübergang.",
        "sind Unwetterauswirkungen.",
        "ist verspätetes Personal aus vorheriger Fahrt.",
        "ist die Verspätung eines vorausfahrenden Zuges.",
    ]
    fake_reasons = [
        "ist die Sichtung eines unbekannten Flugobjekts auf der Strecke.",
        "ist ein Geister-Zug auf der Strecke.",
        "ist ein fehlender Bahnhof auf der Strecke.",
        "ist die Verspätung eines nachfolgenden Zuges.",
        "ist die Überschwemmung des Bordrestaurants.",
        "ist ein Notarzteinsatz in Nordrhein-Westfalen.",
        "ein mysteriös verschwundener Wagen.",
        "ist ein geplatzter Reifen.",
        "ist starker Gegenwind.",
        "ist ein Leck an Steuerbord.",
        "ist beschädigter Anker am Zug.",
        "ist eine Baustelle im Zug.",
        "ist der Sommer.",
        "ist ein umgestürzter Baumkuchen auf der Strecke.",
        "ist ein Unfall in der Zugtoilette.",
        "ist ein Maulwurf auf der Strecke.",
        "ist eine eingestürzte Brücke auf der Strecke.",
        "ist eine Umleitung wegen eines eingestürzten Tunnels.",
        "ist ein Vogelschlag.",
        "ist ein Telefongespräch des Zugführers.",
        "ist eine technische Untersuchung an einem Reisenden.",
        "ist ein defektes Mobiltelefon.",
        "ist ein Feuerwehreinsatz im Bordrestaurant.",
        "ist eine Zoll-Erhöhung.",
        "ist ein Zwischenhalt zum Zustieg des Schwagers der Zugbegleiterin.",
        "ist eine abgefallene Ein- und Ausstiegstüre.",
        "ist die Betätigung des Nothalt-Knopfes durch einen Fahrgast.",
        "ist die fehlende Bereitschaft eines Fahrgastes, zuzusteigen.",
        "ist ein Meteoriteneinschlag auf er Strecke.",
        "ist der Fund einer Fliegerbombe.",
        "ist eine Toilettenpause.",
        "ist ein Fahrrad in Wagen drei.",
        "sind Verzögerung beim Ausrollen eines roten Teppichs für den Bürgermeister.",
        "ist eine Signalstörungs-Behebungs-Planungs-Besprechung am Gleis.",
        "ist eine Verzögerung bei der Untersuchung von Stellwerk-Störungen.",
        "ist eine Blaskapelle.",
        "ist eine vorübergehenden Sperrung aller Bordtoiletten.",
        "ist die verfrühte Bereitstellung des Zuges.",
        "ist eine Oberleitungsanordnung.",
        "ist die tierärztliche Versorgung eines an Bord befindlichen Haustiers.",
        "ist die Landung eines Passagierflugzeugs auf der Strecke.",
        "ist fehlendes Toilettenpapier wegen Hamsterkäufen.",
        "ist die verspätete Pizza-Lieferung des Zugführers.",
        "der verlorene Geldbeutel eines Mitreisenden.",
        "die blendende Sonne.",
        "ist ein Tippfehler auf dem Fahrplan.",
        "ist eine umstrittene Aussage des Bundeskanzlers.",
        "ist die Beschädigung einer Kommunikationsleitung.",
        "ist ein Marderschaden.",
        "ist ein Defekt an der Klimaanlage.",
        "sind archäologische Ausgrabungen.",
        "ist die Warnung eines Hellsehers.",
        "ist eine Taube auf dem Zug.",
    ]
    if train in TARGETS:
        name, target = TARGETS[train][platform]
        hour, minute, delay = delayed_now(max(5, delay_minutes))
        reasons = fake_reasons if random.random() < .3 else real_reasons
        of = {S: 'der', E_RB: 'der'}.get(train, 'des')
        exit_type = "Abfahrt" if entered_seconds_ago > 45 else "Weiterfahrt"
        speech = f"Bitte beachten Sie: Die {exit_type} {of} {name} verzögert sich um circa {delay} Minuten. Grund dafür " + random.choice(reasons)
    else:
        speech = random.choice(sentences)
    play_announcement(speech, left_vol=int(platform <= 3), right_vol=int(platform>3))
    return 15


def delayed_now(delay_minutes: int):
    dt = datetime.now()
    delay_minutes = (delay_minutes // 5) * 5
    minutes = round(dt.minute / 5) * 5
    if minutes >= 60:
        dt += timedelta(hours=1)
        minutes = 0
    minute_text = {
        0: "",
        5: "fünf",
        10: "zehn",
        15: "fünfzehn",
        20: "zwanzig",
        25: "fünfundzwanzig",
        30: "dreißig",
        35: "fünfunddreißig",
        40: "vierzig",
        45: "fünfundvierzig",
        50: "fünfzig",
        55: "fünfundfünfzig",
    }
    dt = dt.replace(minute=minutes, second=0, microsecond=0) - timedelta(minutes=delay_minutes)
    return dt.hour, minute_text[dt.minute], delay_minutes


# ToDo sounds only if enabled
READY_SOUNDS = {  # (name, duration, volume)
    ICE: ("whistle1.wav", 1.5, 1.),
    S: ("door-beep-S-Bahn.wav", 3.5, 1.),
    SHUTTLE: ("door-beep-RE.wav", 5., 1.),
    E_BW: ("whistle2.wav", 1.5, 1.),
    E_RB: ("door-beep-RE.wav", 5., 1.),
    DAMPF: ("steam-horn.wav", 3.5, 1.),  # oder Horn vom Zug
    E40: ("whistle-and-train1.wav", 1.5, .2),
    BEIGE: ("diesel-steam.wav", 0., 1.),
    ROT: ("diesel-steam.wav", 2., 1.),
    DIESEL: ("diesel-steam.wav", 2., 1.),
    BUS: ("doors-tram.wav", 1.5, 1.),
}

DEPARTURE_SOUNDS = {  # (filename, volume)
    ICE: ("e-train.mp3", .3),
    # S: None,  # sound from train
    E_BW: ("e-train.mp3", .3),
    E_RB: ("e-train.mp3", .3),
    # DAMPF: None,  # sound from train
    E40: ("e-train.mp3", .3),
    BEIGE: ("diesel-departure.mp3", 1.),
    # ROT: None,
    DIESEL: ("diesel-departure.mp3", 1.),
    BUS: ("tram.mp3", 1.),
    # SHUTTLE: ("e-train.mp3", .3),
}


MEASUREMENT_DATA = []


def write_measurement(name: str, platform: int, speed_level: int, time_since_trip: float, time_since_clear: float):
    logger = logging.getLogger(__name__)
    filename = 'terminus-measurements.json'
    if not MEASUREMENT_DATA:
        if os.path.isfile(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                MEASUREMENT_DATA.extend(json.load(f))
    MEASUREMENT_DATA.append({
        "train": name,
        "platform": platform,
        "speed_level": speed_level,
        "time_since_trip": time_since_trip,
        "time_since_clear": time_since_clear,
    })
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(MEASUREMENT_DATA, f, indent=2)
    logger.info(f"Measurement recorded: {name} on platform {platform}, speed_level={speed_level}, entry_time={time_since_trip:.2f}s, clear_time={time_since_clear:.2f}s")


if __name__ == '__main__':
    # play_departure(ICE)
    # relays = RelayManager()
    # def main(relay: Relay8):
    #     relay.close_channel(ENTRY_POWER)
        # for i in range(100):
            # relay.open_channel(6)
            # relay.open_channel(8)
            # relay.open_channel(7)
            # time.sleep(1)
            # relay.close_channel(7)
            # relay.close_channel(6)
            # relay.close_channel(8)
            # time.sleep(1)
    # relays.on_connected(main)
    # play_connections(2, [(E_RB, 1), (S, 3)])
    # time.sleep(20)

    logger = logging.getLogger(__name__)
    for i in range(10):
        logger.info(f"Test selection: {select_track(E_BW, {i: 'empty' for i in range(1, 6)})}")
