from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Optional

from fpme.audio import async_play
from fpme.train_control import TrainControl


class Scheduler:

    def __init__(self, delays: list[float], target: Callable[[int], None]):
        if not delays:
            raise ValueError("delays must not be empty")
        if any(delay < 0 for delay in delays):
            raise ValueError("delays must be non-negative")
        self._delays = delays
        self._target = target
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._generation = 0
        self._position = 0
        self._deadline: Optional[float] = None
        self.on_start = None
        self.on_stop = None

    def start(self, position: Optional[int] = None) -> None:
        """Start or reset the scheduler.

        position is the index whose delay will be waited before target(index)
        is called. If omitted, scheduling starts at index 0.
        """
        if position is None:
            position = 0
        if not 0 <= position < len(self._delays):
            raise IndexError(f"position must be between 0 and {len(self._delays) - 1}")
        with self._lock:
            # Invalidate the currently scheduled timer.
            self._generation += 1
            generation = self._generation
            if self._timer is not None:
                self._timer.cancel()
            self._position = position
            self._schedule_locked(position, generation)
        self.on_start()

    def stop(self) -> None:
        """Stop the scheduler. Has no effect if it is already stopped."""
        with self._lock:
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._deadline = None
        self.on_stop()

    @property
    def running(self):
        return self._deadline is not None

    @property
    def current_position(self):
        return self._position

    @property
    def time_left(self) -> float:
        with self._lock:
            return -1.0 if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    def _schedule_locked(self, position: int, generation: int) -> None:
        delay = self._delays[position]
        self._deadline = time.monotonic() + delay
        timer = threading.Timer(delay, self._run, args=(position, generation))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _run(self, position: int, generation: int) -> None:
        with self._lock:
            if generation != self._generation:  # Check that this timer is still the current scheduler instance.
                return
            self._timer = None
            self._deadline = None
        self._target(position)  # Execute outside the lock so target() cannot block start()/stop().
        with self._lock:
            if generation != self._generation:  # target() may have called start() or stop().
                return
            next_position = (position + 1) % len(self._delays)
            self._position = next_position
            self._schedule_locked(next_position, generation)


def play_pause_announcement(fake_prob=.5):
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
    reason = random.choice(fake_reasons if random.random() < fake_prob else real_reasons)
    from fpme.audio import play_announcement
    play_announcement(f"Bitte beachten Sie: Der Zugverkehr wird vorübergehend eingestellt. Grund dafür " + reason)


def play_resume_announcement():
    async_play("ansagen/gong3-reverb.wav")


def create_scheduler(control: TrainControl, drive_duration: float, pause_duration: float):
    def reset():
        for train in control.trains:
            state = control[train]
            state.set_speed_limit('pause', None, cause='scheduler')
    def phase_end(phase: int):
        if phase == 0:
            print(">>> Pause <<<")
            # control.power_off(None, cause="Pause")
            for train in control.trains:
                state = control[train]
                is_entering = state.track == 'terminus' and state.speed != 0
                if not is_entering:
                    state.set_speed_limit('pause', 0., jerk=False, cause='scheduler')
            play_pause_announcement()
        else:
            print(">>> Pause Ende <<<")
            reset()
            play_resume_announcement()
    scheduler = Scheduler([drive_duration, pause_duration], phase_end)
    scheduler.on_stop = lambda: reset() or play_resume_announcement()
    scheduler.on_start = lambda: reset() or play_resume_announcement()
    return scheduler


if __name__ == '__main__':
    s = Scheduler([5, 1], lambda i: print(f"{i} ended"))
    print(s.time_left, s.current_position, s.running)
    print("Starting with 0")
    s.start(0)
    time.sleep(1)
    print(s.time_left, s.current_position, s.running)
    time.sleep(100)