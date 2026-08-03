"""
Requires VR-Park controllers to be in mode C.

Cross-platform unlike winusb. Both can read messages from VR-Park controllers in mode C.
"""
import sys
import time
import traceback
import warnings
from functools import partial
from threading import Thread
from typing import List, Dict, Optional, Set, Tuple

import pywinusb.hid as hid

from fpme.terminus import Terminus, play_special_announcement
from fpme.train_control import TrainControl


class InputManager:

    def __init__(self, control: Optional[TrainControl]):
        self.control = control
        self.terminus = None
        self.connected: Dict[str, Optional[hid.HidDevice]] = {}
        self.disconnected: Set[str] = set()
        self.last_events: Dict[str, Tuple[float, str]] = {}  # (time, text)
        self.button_states: Dict[str, Dict[str, Tuple[bool, float]]] = {}  # (pressed, time_last_pressed)
        self.check_for_new_devices(auto_activate=False)

    def set_terminus(self, terminus: Terminus):
        self.terminus = terminus

    def check_for_new_devices(self, auto_activate=True):  # 1452, 556
        devices = hid.find_all_hid_devices()
        controllers = {dev.device_path: dev for dev in devices if is_controller(dev)}
        # --- Remove disconnected controllers ---
        for path in tuple(self.connected):
            if path not in controllers:
                print(f"Controller disconnected: {path}")
                del self.connected[path]
                self.disconnected.add(path)
                if self.control is not None:
                    self.control.remove_controller(path)
        # --- Add new controllers ---
        for device in [dev for path, dev in controllers.items() if path not in self.connected]:
            try:
                device.open()
                print(f"Opened new controller: {device.product_name} @ {device.device_path}")
                device.set_raw_data_handler(partial(self.process_event, device_path=device.device_path))
                if device.device_path in self.disconnected:
                    self.disconnected.remove(device.device_path)
                self.last_events[device.device_path], self.connected[device.device_path] = (time.perf_counter(), 'connected'), device
                if device.device_path in CONTROLS:
                    train = CONTROLS[device.device_path]
                    if auto_activate and self.control is not None:
                        self.control.activate(train, device.device_path)
                else:
                    print("This controller has not been assigned to any train! Copy the following Python path")
                    print("'" + device.device_path.replace("\\", "\\\\") + "'")
            except OSError as e:
                print(f"Failed to open bluetooth controller: {e}")
                self.connected[device.device_path] = None

    def start_detection(self, interval_sec=1.):
        def detection_loop():
            while True:
                self.check_for_new_devices()
                time.sleep(interval_sec)
        Thread(target=detection_loop).start()

    def process_event(self, data: List, device_path: str):
        try:
            t = time.perf_counter()
            train = CONTROLS.get(device_path)
            device_name = self.connected[device_path].product_name if device_path in self.connected else device_path
            if device_name == "@input.inf,%hid_device_system_game%;HID-compliant game controller":
                x, acc, buttons = get_vr_park_state(data)
                bindings = VR_PARK_BIND
            elif device_name == "Twin USB Joystick":
                x, acc, buttons = get_twin_joystick_state(data)
                bindings = TWIN_JOYSTICK_BIND
            else:
                warnings.warn(f"Unknown input device: {device_name} @ {device_path}")
                self.last_events[device_path] = (time.perf_counter(), int_list_to_binary_visual(data))
                return
            # --- Evaluate presses, double clicks ---
            prev = self.button_states.setdefault(device_path, {b: (False, -1) for b in buttons})
            self.button_states[device_path] = {b: (p, t if p else self.button_states[device_path][b][1]) for b, p in buttons.items()}
            presses = {b: 'double' if t - prev[b][1] < 0.4 else 'press' for b, p in buttons.items() if p and not prev[b][0]}
            actions = {bindings[b]: p for b, p in presses.items()}  # e.g. {'stop': 'press'}
            # --- Apply control ---
            if self.control is not None and train is not None:
                self.control.set_acceleration_control(train, device_path, acc, cause=device_path)
                if self.terminus:
                    self.terminus.correct_move(train, x if acc == 0 else 0)
                if 'reverse' in actions:
                    if actions['reverse'] == 'press':
                        self.control.reverse(train, cause=device_path, emergency_stop=True)
                        if self.terminus:
                            self.terminus.on_reversed(train)
                    else:  # 'double'
                        self.control.emergency_stop_all(train, cause=device_path)
                if 'terminus' in actions:
                    if self.terminus:
                        if actions['terminus'] == 'press':
                            self.terminus.request_entry(train)
                        else:
                            self.terminus.remove_train(train)
                    else:
                        print("no terminus set")
                for fun_i, fun in TRAIN_FUNCTIONS.items():
                    if fun in actions:
                        if actions[fun] == 'press':
                            self.control.use_ability(train, fun_i, cause=device_path, check_cooldown=True)
                        else:  # double-click any function to restore power
                            self.control.reset_power(train, cause=train.name, stop_trains=True)
            # --- Remember event ---
            if presses:
                event_text = ','.join([f"{a}-{p} ({b})" for (b, p), a in zip(presses.items(), actions)])
                self.last_events[device_path] = (time.perf_counter(), event_text)
            elif acc:
                event_text = f'a={acc}'
                self.last_events[device_path] = (time.perf_counter(), event_text)
        except BaseException as err:  # make sure the event loop stays active
            traceback.print_exc()


TRAIN_FUNCTIONS = {0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4'}


def is_controller(dev: hid.HidDevice):
    if dev.product_name == "@input.inf,%hid_device_system_game%;HID-compliant game controller":
        return 'vid&0205ac_pid&022c' in dev.device_path  # VR Park (Bluetooth)
    elif dev.product_name == "Twin USB Joystick":
        return 'col01' in dev.device_path  # Controller (wireless USB)
    return False


def get_twin_joystick_state(data: List) -> Tuple[float, float, Dict[str, bool]]:
    # assume in mode RED
    left_y = data[4]  # 0 up, 128 center, 255 down  left joystick or up/down buttons
    left_x = data[3]  # 0 left, 128 center, 255 right  left joystick or up/down buttons
    rlb_pressed = data[6] & 3  # reverse: either left or right button (upper trigger)
    rlt_pressed = data[6] & 12  # stop: either left or right lower trigger
    y_pressed = data[5] & 16  # Terminus
    a_pressed = data[5] & 64  # F1
    x_pressed = data[5] & 128  # F2
    b_pressed = data[5] & 32  # F3
    state = {'A': a_pressed, 'B': b_pressed, 'X': x_pressed, 'Y': y_pressed, 'R/LT': rlt_pressed, 'R/LB': rlb_pressed}
    y = {0: 1., 128: 0., 255: -1.}[left_y]
    x = {0: -1., 128: 0., 255: 1.}[left_x]
    return x, y, state


TWIN_JOYSTICK_BIND = {
    'A': 'F2',
    'B': 'F3',
    'X': 'F1',
    'Y': 'terminus',
    'R/LT': 'reverse',
    'R/LB': 'reverse',
}


def get_vr_park_state(data: List) -> Tuple[float, float, Dict[str, bool]]:
    # data is always [4, 127, 127, 127, 128, button_id, 0, hat, 0]
    _, _, _, _, _, pressed, _, hat, _ = data
    hat_pos = {
        0: (0, 0),
        7: (0, 1),
        3: (0, -1),
        1: (1, 0),
        5: (-1, 0),
        8: (1, 1),
        2: (1, -1),
        4: (-1, -1),
        6: (-1, 1),
    }[hat]
    state = {'A/T': pressed == 16, 'B/B': pressed == 1, 'C': pressed == 8, 'D': pressed == 2}
    return *hat_pos, state


VR_PARK_BIND = {
    'A/T': 'reverse',
    'B/B': 'F1',
    'C': 'terminus',
    'D': 'F2',
}


def int_list_to_binary_visual(numbers):
    """Convert a list of integers [0, 255] to 8-bit binary representation using X for 1 and _ for 0."""
    binary_codes = []
    for num in numbers:
        # Convert to 8-bit binary string
        binary_str = f"{num:08b}"
        # Replace 1 with X and 0 with _
        visual_binary = binary_str.replace('1', 'X').replace('0', '_')
        binary_codes.append(visual_binary)
    return ", ".join(binary_codes)


from fpme.train_def import *
CONTROLS = {
    '\\\\?\\hid#vid_0810&pid_0001&col01#8&3a1c6485&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}': ROT,
    '\\\\?\\hid#vid_0810&pid_0001&col01#8&14401120&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}': ROT,
    '\\\\?\\hid#vid_0810&pid_0001&col01#8&186607b&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}': DAMPF,
    '\\\\?\\hid#vid_0810&pid_0001&col01#8&598cc1&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}': DAMPF,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_e0eebb488ace&col04#b&316fd37&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': ROT,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_fdc80171a4bd&col04#b&2a97252d&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': ICE,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_1e89fff2c7db&col04#b&20fc5261&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': BUS,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_588ea725c7a3&col04#b&13df0efa&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': BEIGE,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_45ee447eb09a&col04#b&20a8da1a&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': DIESEL,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_3675eb0ae2f9&col04#b&29342c48&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': S,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_8fd1f0ddfca6&col04#b&2671a960&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': E_RB,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_61f5dd3d7341&col04#b&2c7afc78&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': E40,
    '\\\\?\\hid#{00001812-0000-1000-8000-00805f9b34fb}_dev_vid&0205ac_pid&022c_rev&011b_9b64950eee81&col04#b&1cbc610d&0&0003#{4d1e55b2-f16f-11cf-88cb-001111000030}': E_BW,
}

if __name__ == '__main__':
    inputs = InputManager(None)
    inputs.start_detection()