import sys
import os

from fpme.schedule import create_scheduler
from fpme.train_control import TrainControl
from fpme.train_def import TRAINS, GUETER

if __name__ == '__main__':
    from fpme.hid_input import InputManager
    from fpme.relay8 import RelayManager, Relay8
    from fpme.terminus import Terminus  # this imports PyGame

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from fpme import tk_gui, signal_gen

    control = TrainControl()
    for train in TRAINS:
        if train.info == GUETER:
            control[train].track = 'regional'
    control.load_state()
    ports = [p for p in signal_gen.list_com_ports(include_bluetooth=False) if 'Prolific' in p[1]]
    for port, desc, _ in ports:
        control.add_rs232_generator(port)
    if not control.generator.get_open_ports():
        control.add_rs232_generator('debug1:on')
        # control.add_rs232_generator('debug2:off')
    control.power_on(None, 'launch')
    inputs = InputManager(control)
    inputs.start_detection()
    relay = RelayManager()
    scheduler = create_scheduler(control, drive_duration=6.5 * 60, pause_duration=30.)
    # Thread(target=lambda: dash_app.Server(control).launch(port=PORT)).start()
    gui = tk_gui.TKGUI(control, scheduler, relay, inputs, infos=[], fullscreen=False)

    def setup_terminal(relay: Relay8):
        print("Relay detected. Setting up terminus.")
        terminus = Terminus(relay, control, ports[0][0])
        terminus.reverse_to_exit()
        gui.set_terminus(terminus)
        inputs.set_terminus(terminus)
    relay.on_connected(setup_terminal)

    gui.launch()
