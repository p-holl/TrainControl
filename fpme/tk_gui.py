import math
import os
import time
import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict, List, Optional

from PIL import ImageTk, Image

from .helper import fit_image_size
from .hid_input import InputManager, CONTROLS
from .relay8 import RelayManager
from .signal_gen import list_com_ports
from .terminus import Terminus
from .train_control import TrainControl
from .train_def import Train, obstacle, MW_TGV


class TKGUI:

    def __init__(self, control: TrainControl, relays: RelayManager, inputs: InputManager, infos=(), fullscreen=False):
        self.control = control
        self.relays = relays
        self.terminus = None
        self.selected_platform = None
        self.inputs = inputs
        self.window = tk.Tk()
        self.speed_bars: Dict[Train, ttk.Progressbar] = {}
        self.direction_labels: Dict[Train, tk.Label] = {}
        self.shown_trains: List[Train] = []

        self.window.title("Modellbahn Steuerung")
        self.window.geometry('800x750')
        # tk.Label(text="Press F11 to enter fullscreen mode").pack()
        if fullscreen:
            self.window.attributes("-fullscreen", True)
        for info in infos:
            tk.Label(text=info).pack()
        status_pane = tk.Frame(self.window)
        status_pane.pack()
        # --- Hardware ---
        tk.Label(status_pane, text="Hardware", font='Helvetica 14 bold').grid(row=0, column=0)
        hardware_pane = tk.Frame(status_pane)
        hardware_pane.grid(row=1, column=0)
        self.status_labels = {}  # port -> Label
        row = 0
        port_descriptions = {port: desc for port, desc, hwid in list_com_ports(include_bluetooth=True)}
        for port in control.generator.get_open_ports():
            tk.Label(hardware_pane, text=port).grid(row=row, column=0)
            tk.Label(hardware_pane, text=port_descriptions.get(port, 'Fake port')).grid(row=row, column=1)
            status_label = tk.Label(hardware_pane, text="unknown")
            status_label.grid(row=row, column=2)
            self.status_labels[port] = status_label
            row += 1
        # --- Relay ---
        tk.Label(hardware_pane, text="Terminus Relay").grid(row=row, column=0)
        tk.Label(hardware_pane, text="USB switch control").grid(row=row, column=1)
        status_label = tk.Label(hardware_pane, text="unknown")
        status_label.grid(row=row, column=2)
        self.status_labels['terminus'] = status_label
        row += 1
        # --- Event highlights ---
        tk.Label(status_pane, text="Events", font='Helvetica 14 bold').grid(row=0, column=1)
        event_pane = tk.Frame(status_pane)
        event_pane.grid(row=1, column=1)
        self.emergency_break_all_highlight = tk.Label(event_pane, text="")
        self.emergency_break_all_highlight.pack()
        self.power_off_highlight = tk.Label(event_pane, text="Power off")
        self.power_off_highlight.pack()
        self.power_on_highlight = tk.Label(event_pane, text="Power on")
        self.power_on_highlight.pack()
        # --- Trains ---
        # tk.Label(text="Controls", font='Helvetica 14 bold').pack()
        controls_pane = tk.Frame(self.window)
        controls_pane.pack()
        self.last_action_labels = {}
        self.photos = []
        self.track_labels = {}
        self.train_images = {}
        def add_progress_bar(train: Train):
            progress_bar = ttk.Progressbar(controls_pane, value=50, length=100)
            progress_bar.grid(row=row, column=4)
            self.speed_bars[train] = progress_bar
            direction_label = tk.Label(controls_pane, text='')
            direction_label.grid(row=row, column=5)
            self.direction_labels[train] = direction_label
        for device_path, train in CONTROLS.items():
            row = self.control.trains.index(train)
            self.shown_trains.append(train)
            photo = ImageTk.PhotoImage(train.image.resize(fit_image_size(train.img_res, 80, 30)))
            photo_bw = ImageTk.PhotoImage(greyscale(train.image.resize(fit_image_size(train.img_res, 80, 30))))
            self.photos.extend([photo, photo_bw])
            image_label = tk.Label(controls_pane, text=train.name, image=photo, compound=tk.LEFT)
            image_label.grid(row=row, column=2)
            self.train_images[train] = (image_label, photo, photo_bw)
            if train in control.trains:
                add_progress_bar(train)
            else:
                tk.Label(controls_pane, text="not managed").grid(row=row, column=3)
            last_action_label = tk.Label(controls_pane, text='nothing')
            last_action_label.grid(row=row, column=1)
            last_action_label.config(width=16, height=2)
            self.last_action_labels[device_path] = last_action_label
            row += 1
        for train in control.trains:
            row = self.control.trains.index(train)
            if train not in CONTROLS.values():
                tk.Label(controls_pane, text=train.name).grid(row=row, column=2)
                add_progress_bar(train)
                row += 1
                self.shown_trains.append(train)
            track = tk.Label(controls_pane, text="?")
            track.grid(row=row, column=6)
            self.track_labels[train] = track
        # --- Status ---
        tk.Label(status_pane, text="State", font='Helvetica 14 bold').grid(row=0, column=2)
        # tk.Label(text="Status (P/R)", font='Helvetica 14 bold').pack()
        state_pane = tk.Frame(status_pane)
        state_pane.grid(row=1, column=2)
        tk.Label(state_pane, text="Status (P/R)").grid(row=0, column=0)
        self.active_status = tk.Label(state_pane, text="...")
        self.active_status.grid(row=0, column=1)
        tk.Label(state_pane, text="Light (F2/F3)").grid(row=1, column=0)
        self.light_status = tk.Label(state_pane, text="...")
        self.light_status.grid(row=1, column=1)
        tk.Label(state_pane, text="Sound (F6/F7)").grid(row=2, column=0)
        self.sound_status = tk.Label(state_pane, text="...")
        self.sound_status.grid(row=2, column=1)
        tk.Label(state_pane, text="Limit (+/-)").grid(row=3, column=0)
        self.speed_limit = tk.Label(state_pane, text="...")
        self.speed_limit.grid(row=3, column=1)
        # --- Terminus ---
        terminus_pane = tk.Frame(self.window)
        terminus_pane.pack()
        self.canvas = tk.Canvas(terminus_pane, width=800, height=300)
        self.canvas.pack()
        terminus_img = Image.open("assets/Kopfbahnhof final.jpg")
        terminus_img = terminus_img.resize((800, 300))
        photo_image = ImageTk.PhotoImage(terminus_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.sel_platform = self.canvas.create_rectangle(0, 0, 300, 10, fill='blue')
        obs = obstacle()
        self.canvas_images = {'__bg__': photo_image, 'Baustelle': ImageTk.PhotoImage(obs.image.resize(fit_image_size(obs.img_res, 80, 30)))}  # Keep a reference to images to prevent garbage collection
        self.canvas_ids = {}
        self.canvas_texts = {}
        for train in control.all_trains:
            train_photo = self.canvas_images[train] = ImageTk.PhotoImage(train.image.resize(fit_image_size(train.img_res, 80, 30)))
            self.canvas_ids[train] = self.canvas.create_image(0, 0, anchor=tk.NW, image=train_photo)
            self.canvas_texts[train] = self.canvas.create_text(0, 0, anchor=tk.NW)
        # --- Hotkeys ---
        self.window.bind("<F11>", lambda e: self.window.attributes("-fullscreen", not self.window.attributes('-fullscreen')))
        self.window.bind("<F2>", lambda e: self.control.set_lights_on(False))
        self.window.bind("<F3>", lambda e: self.control.set_lights_on(True))
        self.window.bind("<F6>", lambda e: self.control.set_sound_on(False))
        self.window.bind("<F7>", lambda e: self.control.set_sound_on(True))
        self.window.bind("<p>", lambda e: self.control.pause())
        self.window.bind("<r>", lambda e: self.control.resume())
        self.window.bind("<+>", lambda e: self.control.set_global_speed_limit(None if self.control.speed_limit is None else (self.control.speed_limit + 20 if self.control.speed_limit < 240 else None)))
        self.window.bind("<minus>", lambda e: self.control.set_global_speed_limit(240 if self.control.speed_limit is None else self.control.speed_limit - 20))
        self.window.bind("<Escape>", lambda e: self.terminate())
        self.window.bind("1", lambda e: self.terminus_set(0))
        self.window.bind("2", lambda e: self.terminus_set(1))
        self.window.bind("3", lambda e: self.terminus_set(2))
        self.window.bind("4", lambda e: self.terminus_set(3))
        self.window.bind("5", lambda e: self.terminus_set(4))
        self.window.bind("6", lambda e: self.terminus_set(5))
        self.window.bind("7", lambda e: self.terminus_set(6))
        self.window.bind("8", lambda e: self.terminus_set(7))
        self.window.bind("9", lambda e: self.terminus_set(8))
        self.window.bind("0", lambda e: self.terminus_set(9))
        self.window.bind("<o>", lambda e: self.terminus_set(-1))
        self.window.bind("<Control-Key-1>", lambda e: self.terminus_select(1))
        self.window.bind("<Control-Key-2>", lambda e: self.terminus_select(2))
        self.window.bind("<Control-Key-3>", lambda e: self.terminus_select(3))
        self.window.bind("<Control-Key-4>", lambda e: self.terminus_select(4))
        self.window.bind("<Control-Key-5>", lambda e: self.terminus_select(5))
        self.window.bind("<BackSpace>", lambda e: self.clear_platform())
        self.window.bind("<d>", lambda e: print(self.control.generator.format_state()))
        self.window.bind("<Return>", lambda e: self.external_train_enter())
        self.window.protocol("WM_DELETE_WINDOW", lambda: self.terminate())

    def launch(self):
        self.window.after(10, self.update_ui)
        self.window.mainloop()

    def set_terminus(self, terminus: Terminus):
        self.terminus = terminus
        for t in terminus.trains:
            if t.train not in self.canvas_ids:
                self.canvas_ids[t.train] = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_images["Baustelle"])

    def terminus_select(self, platform: int):
        if self.terminus is not None:
            self.selected_platform = platform

    def terminus_set(self, train_id: Optional[int]):
        if train_id >= len(self.shown_trains):
            return
        if self.selected_platform is None:
            return
        if train_id == -1:  # Obstacle
            train = obstacle()
            self.canvas_ids[train] = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_images["Baustelle"])
        else:
            train = self.control.trains[train_id]
        if self.terminus is not None:
            self.terminus.set_occupied(self.selected_platform, train)
        self.selected_platform = None

    def clear_platform(self):
        self.terminus.set_empty(self.selected_platform)
        self.selected_platform = None

    def external_train_enter(self, train=MW_TGV):
        if not self.terminus:
            return
        if self.terminus.is_in_terminus(train):
            self.terminus.remove_train(train)
        else:
            self.terminus.request_entry(train)

    def update_ui(self):
        now = time.perf_counter()
        # --- Highlight recent inputs and detect disconnected devices ---
        for device, (t, text) in dict(self.inputs.last_events).items():
            if device in self.last_action_labels:
                label = self.last_action_labels[device]
                if device in self.inputs.disconnected:
                    label.config(text='disconnected', bg=tk_rgb(255, 0, 0))
                else:
                    fac = int(255 * (1 - math.exp(t - now)))
                    if not (fac >= 0 and fac <= 255):
                        fac = 100
                    label.config(text=text, bg=tk_rgb(fac, 255, fac))
        # -- Highlight recent global commands ---
        cause_text = lambda x: CONTROLS[x].name if x in CONTROLS else x
        fac = 1 - math.exp(self.control.last_emergency_break_all[0] - now)
        self.emergency_break_all_highlight.config(text=f"Emergency all: {cause_text(self.control.last_emergency_break_all[1])}", bg=tk_rgb(255, int(255 * fac), int(255 * fac)))
        fac = 1 - math.exp(self.control.last_power_off[0] - now)
        self.power_off_highlight.config(text=f"Power off: {cause_text(self.control.last_power_off[1])}", bg=tk_rgb(255, int(255 * fac), int(255 * fac)))
        fac = 1 - math.exp(self.control.last_power_on[0] - now)
        self.power_on_highlight.config(text=f"Power on: {cause_text(self.control.last_power_on[1])}", bg=tk_rgb(int(255 * fac), 255, int(255 * fac)))
        # --- Update train display ---
        for train in self.control.trains:
            self.speed_bars[train].config(value=abs(100 * (self.control[train].speed or 0.) / train.max_speed))
            self.direction_labels[train].config(text='🡄' if self.control[train].is_in_reverse else '🡆')
        for train, (label, img, img_bw) in self.train_images.items():
            label.config(image=img if self.control[train].is_active else img_bw)
        for train, label in self.track_labels.items():
            track = self.control[train].track
            label.config(text="?" if track is None else track)
        # --- Update status displays ---
        for port in self.control.generator.get_open_ports():
            error = self.control.generator.get_error(port)
            if error:
                signal_status = f"⛔ {error}"
            elif self.control.generator.is_short_circuited(port):
                signal_status = "⚠ short-circuited or no power"
            elif self.control.generator.is_sending_on(port):
                signal_status = "✅"
            elif self.control.paused:
                signal_status = "paused"
            else:
                signal_status = "⚠"
            self.status_labels[port].config(text=signal_status)
        # --- Relay ---
        terminus_status = "✅" if self.relays.is_connected else " ⛔ " + self.relays.status
        self.status_labels['terminus'].config(text=terminus_status)
        # --- State ---
        self.active_status.config(text="paused" if self.control.paused else "not paused")
        self.light_status.config(text="on" if self.control.light else ("?" if self.control.light is None else "off"))
        self.sound_status.config(text=["off", "station", "all"][self.control.sound])
        self.speed_limit.config(text=str(self.control.speed_limit))
        # --- Terminus plan ---
        if self.terminus:
            self.canvas.pack()
            for train, img_id in self.canvas_ids.items():
                platform, pos = self.terminus.get_train_position(train)
                if pos is not None:
                    pos -= 20  # 0 is at track outside board
                if platform:
                    y = {1: 15, 2: 65, 3: 115, 4: 170, 5: 220}[platform]
                    x = pos * (800/250) + 50 if pos is not None else 10
                    x = max(-20, min(x, 750))
                    label = format(pos, '.1f') if pos is not None else "?"
                else:
                    x, y = -100, -100
                    label = ""
                self.canvas.coords(img_id, x, y)
                if train in self.canvas_texts:
                    self.canvas.coords(self.canvas_texts[train], x, y)
                    self.canvas.itemconfig(self.canvas_texts[train], text=label)
            # --- Highlight platform selection ---
            if self.selected_platform is None:
                self.canvas.coords(self.sel_platform, -100, -100, 1, 1)
            else:
                y = {1: 12, 2: 68, 3: 118, 4: 177, 5: 224}[self.selected_platform]
                self.canvas.coords(self.sel_platform, 600, y, 800, y+10)
        else:
            self.canvas.pack_forget()
        # --- Schedule next update ---
        self.window.after(10, self.update_ui)

    # def toggle_active(self, train_id: int):
    #     if train_id >= len(self.shown_trains):
    #         return
    #     train = self.shown_trains[train_id]
    #     is_active = self.control[train].is_active
    #     if is_active:
    #         self.control.deactivate(train, "UI")
    #     else:
    #         self.control.activate(train, "UI")

    def terminate(self):
        self.control.generator.terminate()
        self.control.save_state()
        time.sleep(.5)
        if self.terminus:
            self.terminus.save_state()
        os._exit(0)


def tk_rgb(r, g, b):
    return "#%02x%02x%02x" % (r, g, b)


def greyscale(img: Image):
    r, g, b, a = img.split()
    grey = Image.merge("RGB", (r, g, b)).convert('L')
    return Image.merge("LA", (grey, a)).convert('RGBA')