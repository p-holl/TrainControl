import os
import time
from threading import Thread
from typing import Union

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

import pygame
import pyttsx3


pygame.mixer.init()
engine = pyttsx3.init()

DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/sound"))
_, ir = wavfile.read(DIR + "/ansagen/IR-terminus.wav")
if ir.ndim > 1:
    ir = np.mean(ir, axis=1)
ir = ir / np.max(np.abs(ir))

GONG = pygame.mixer.Sound(DIR + "/ansagen/gong-reverb.wav")
ANSCHLUESSE = pygame.mixer.Sound(DIR + "/ansagen/anschluesse-reverb.wav")


def play_background_loop(file: str):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play(loops=-1)  # infinite loop


def set_background_volume(volume):
    pygame.mixer.music.set_volume(volume)


def async_play(sound: Union[str, pygame.mixer.SoundType], left_vol=1., right_vol=1.):
    if isinstance(sound, str):
        sound = pygame.mixer.Sound(DIR + "/" + sound)
    channel = pygame.mixer.find_channel()
    channel.set_volume(left_vol, right_vol)  # Left speaker only
    channel.play(sound)


def play_announcement(text: str, language='German', speed=160, left_vol=1., right_vol=1., cue='gong', cue_vol=.4):
    if not text:
        return
    Thread(target=_play_announcement, args=(text, language, speed, left_vol, right_vol, cue, cue_vol)).start()


def _play_announcement(text: str, language: str, speed: int, left_vol: float, right_vol: float, cue: str, cue_vol: float):
    t0 = time.perf_counter()
    if cue == 'gong':
        async_play(GONG, left_vol=left_vol * cue_vol, right_vol=right_vol * cue_vol)
        cue_duration = 1.9
    elif cue == 'anschlüsse':
        async_play(ANSCHLUESSE, left_vol=left_vol * cue_vol, right_vol=right_vol * cue_vol)
        cue_duration = 1.9
    else:
        cue_duration = 0
    # --- Generate speech ---
    voices = engine.getProperty('voices')
    german_voices = [voice for voice in voices if language in voice.name]
    if german_voices:
        engine.setProperty('voice', german_voices[0].id)
    else:
        print("No German voice found. Using default voice.")
    engine.setProperty('rate', speed)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    if not os.path.exists('output'):
        os.makedirs('output')
    engine.save_to_file(text, DIR + "/speech.wav")
    engine.runAndWait()
    # --- Reverb and play ---
    apply_reverb("speech.wav", "speech-reverb.wav")
    sound = pygame.mixer.Sound(DIR + "/speech-reverb.wav")  # pre-load sound file
    time.sleep(max(0, cue_duration - t0 + time.perf_counter()))  # wait for gong to subside
    async_play(sound, left_vol=left_vol, right_vol=right_vol)


def apply_reverb(file: str, output_file: str):
    sr, sound = wavfile.read(DIR + "/" + file)
    if sound.ndim > 1:
        sound = np.mean(sound, axis=1)
    wet = fftconvolve(sound, ir, mode='full')  # Apply convolution reverb
    wet = wet / np.max(np.abs(wet))
    wet = (wet * 32767).astype(np.int16)
    wavfile.write(DIR + "/" + output_file, sr, wet)


if __name__ == '__main__':
    # play_announcement("Ihre nächsten Anschlüsse:", cue_vol=1)  # stores speech in assets/sound/speech-reverb.wav
    # apply_reverb("ansagen/gong.wav", "ansagen/gong-reverb-reduced.wav")

    play_announcement("S 1 nach Radeburg von Gleis 5", cue='anschlüsse', cue_vol=1)
    time.sleep(100)
