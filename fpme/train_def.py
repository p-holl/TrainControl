import os
import random
from functools import cached_property
from typing import Tuple, Sequence, Optional, List, Dict

import numpy as np
from PIL import Image
from dataclasses import dataclass, field

TAG_DEFAULT_LIGHT = 'default-light'
TAG_DEFAULT_SOUND = 'default-sound'
TAG_SPECIAL_LIGHT = 'special-light'
TAG_SPECIAL_SOUND = 'special-sound'


@dataclass(frozen=True)
class TrainFunction:
    name: str
    id: int
    default_status: bool
    tags: Tuple[str]
    cooldown: float = 30.
    default_duration: Optional[float] = 1.1

    def __repr__(self):
        return self.name + ":" + str(self.id)


LIGHT = TrainFunction('Licht', 0, False, (TAG_DEFAULT_LIGHT,))
SLOW_MODE = TrainFunction("Langsam-Modus", 3, False, ())
INSTANT_ACCELERATION = TrainFunction("Instantane Beschleunigung", 4, True, ())
SOUND = TrainFunction("Motor", 2, False, (TAG_DEFAULT_SOUND,))


@dataclass(frozen=True)
class TrainInfo:
    name: Optional[str]
    icon: str
    regional_prob: float = .5
    max_delay: int = 60  # arrival delay in minutes
    delay_rate: float = .2
    can_reverse: bool = True
    max_speed_in_station: Tuple[float, float, float] = (50., 50., 60.)  # (multiple switches, single switch, going straight)
    max_speed_by_track: Dict[str, float] = field(default_factory=lambda: {'regional': 160})
    switch_avoidance: float = 0.  # .5 -> 2/5 and is the threshold between 1/3

    def __hash__(self):
        return hash(self.name)

    @cached_property
    def speed_limit_unknown_track(self) -> Optional[float]:
        return None if not self.max_speed_by_track else min(self.max_speed_by_track.values())


@dataclass(frozen=True)
class Train:
    info: TrainInfo
    locomotive: str
    product_number: str
    address: int
    masked_speeds: Sequence[Optional[float]]
    acceleration: float
    custom_deceleration: Optional[float] = None
    has_built_in_acceleration: bool = True
    supports_mm2: bool = True
    stop_by_mm1_reverse: bool = True
    trips_contacts: bool = True
    functions: Tuple[TrainFunction, ...] = (LIGHT,)
    img_path: str = None

    def __post_init__(self):
        assert len(self.masked_speeds) == 15, len(self.masked_speeds)

    @property
    def name(self):
        return self.info.name or ""

    def __repr__(self):
        return self.name or f"{self.locomotive} ({self.info.name})"

    @cached_property
    def speeds(self) -> Tuple[float, ...]:
        return tuple(s for s in self.masked_speeds if s is not None)

    @cached_property
    def speed_codes(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.masked_speeds) if s is not None)

    @cached_property
    def image(self) -> Image:
        return Image.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', self.img_path)) if self.img_path else None

    @property
    def img_res(self):
        if self.image:
            return self.image.size
        else:
            return -1, -1

    @property
    def max_speed(self):
        return self.speeds[-1]

    @cached_property
    def effects(self) -> List[TrainFunction]:
        result = []
        for tag in [TAG_SPECIAL_SOUND, TAG_SPECIAL_LIGHT, TAG_DEFAULT_LIGHT]:
            for f in self.functions:
                if tag in f.tags:
                    result.append(f)
        return result

    @property
    def deceleration(self):
        return self.custom_deceleration if self.custom_deceleration is not None else 2 * self.acceleration

    @property
    def is_passenger_train(self):
        return self.info is not GUETER

    @property
    def id(self):
        return self.locomotive + " / " + self.product_number


def speeds(s15, exponent=1.3):
    return (*np.linspace(0, s15 ** (1/exponent), 15) ** exponent,)


GUETER =            TrainInfo(None, "🚂/🛲", 1, 0, 0, can_reverse=False, max_speed_in_station=(65, 65, 70))
# --- Rail cars ---
ICE_ =              TrainInfo("ICE", "🚅", .33, max_delay=95, delay_rate=.35, max_speed_in_station=(45, 55, 60), switch_avoidance=.9, max_speed_by_track={'regional': 140})
S_ =                TrainInfo("S", "Ⓢ", .75, max_delay=30, delay_rate=.2, max_speed_in_station=(50, 55, 60), switch_avoidance=.5)
BUS_ =              TrainInfo("Bus", "🚌", .8, max_delay=10, delay_rate=0.1, max_speed_in_station=(70, 70, 70))
# --- Wagons ---
INTERREGIO_BLAU =   TrainInfo("IC", "🚉", .3, switch_avoidance=1., max_speed_by_track={'regional': 140})
RB_ROT =            TrainInfo("RB", "🚉", .7, switch_avoidance=.55, max_speed_by_track={'regional': 140})
SILBERLING =        TrainInfo("N/E", "🚉", .6, switch_avoidance=.55, max_speed_by_track={'regional': 140})  # Nahverkehrszug / Eilzug
RE_TUERKIS =        TrainInfo("RE", "🚉", .35, max_speed_in_station=(55, 55, 65), switch_avoidance=.45, max_speed_by_track={'regional': 140})
TGV_ =           TrainInfo(None, "🚂/🛲", .5)


# --- Passenger trains ---
ICE = Train(ICE_, "BR 402 (ICE 2)", "Märklin 29786", 3, speeds(310, 1.4), 25., img_path="ICE.png",
            functions=(LIGHT, SLOW_MODE, INSTANT_ACCELERATION))
E_BW = Train(INTERREGIO_BLAU, "BR 101", "Märklin 37394", 1, (0, 13, 25, 46, 67, 86, 108, 125, 140, 156, 173, 191, 201, 215, 226), 30., img_path="E-Lok BW.png", stop_by_mm1_reverse=False,
             functions=(LIGHT, TrainFunction("Nebelscheinwerfer", 2, False, (TAG_SPECIAL_LIGHT,), 0., None), TrainFunction("Fahrtlicht hinten", 3, False, (TAG_SPECIAL_LIGHT,)), INSTANT_ACCELERATION))
E_RB = Train(RB_ROT, "BR 146.1", "Märklin 29475", 24, speeds(210), 30., supports_mm2=False, stop_by_mm1_reverse=False, img_path="E-Lok DB.png")
S = Train(S_, "BR 648.2 (LINT 41)", "Märklin 37730", 48, (0, 2, 5, 10, 15, 22, 30, 41, 51, 64, 77, 91, 106, 120, 136), 35., img_path="S-Bahn.png",
          functions=(LIGHT, TrainFunction("Innenbeleuchtung", 1, False, (TAG_DEFAULT_LIGHT,)), SOUND, TrainFunction("Horn", 3, False, (TAG_SPECIAL_SOUND,)), INSTANT_ACCELERATION))
BEIGE = Train(SILBERLING, "BR 218", "Märklin 3074", 73, (0, None, 13, 20, 34, 60, 85, 100, 120, 141, 157, 172, 188, 204, 220), 25., img_path="Thumb_BR218_Beige.png",
              functions=(LIGHT, SLOW_MODE, INSTANT_ACCELERATION))
ROT = Train(RE_TUERKIS, 'BR 218', "Märklin 3075", 74, speeds(110), 40., img_path="Thumb_BR218_Rot.png",
            functions=(LIGHT,
                       TrainFunction("Motor", 1, False, (TAG_DEFAULT_SOUND,)),
                       TrainFunction("Horn 1", 3, False, (TAG_SPECIAL_SOUND,)),
                       TrainFunction("Glocke", 2, False, (TAG_SPECIAL_SOUND,)),
                       TrainFunction("Pfeife", 4, False, (TAG_SPECIAL_SOUND,))))
BUS = Train(BUS_, "VT95", "Märklin H0 3016", 62, speeds(190), 40., img_path="Thumb_Schienenbus.png", stop_by_mm1_reverse=False, functions=())

# --- Freight trains ---
DAMPF = Train(GUETER, "BR 81", "Märklin 29185", 78, speeds(210, 1.2), 30., img_path="Dampf.png",
              functions=(LIGHT,
                         TrainFunction("Dampfgeräusche", 1, False, (TAG_DEFAULT_SOUND,)),
                         TrainFunction("Horn", 2, False, (TAG_SPECIAL_SOUND,)),
                         TrainFunction("Glocke", 3, False, (TAG_SPECIAL_SOUND,)),
                         TrainFunction("Überdruck", 4, False, (TAG_SPECIAL_SOUND,))))
DIESEL = Train(GUETER, "V 60 (BR 360)", "Märklin 29155", 72, speeds(180), 25., img_path="Diesel.png",
               functions=(LIGHT, SLOW_MODE, INSTANT_ACCELERATION))
E40 = Train(GUETER, "BR E40", "Märklin 39140", 23, speeds(280, 1.0), 30., img_path="Thumb_E40.png", stop_by_mm1_reverse=False,
            functions=(LIGHT, INSTANT_ACCELERATION, TrainFunction('Horn', 1, False, (TAG_SPECIAL_SOUND,))))  # ToDo which ID is Horn?
MW_TGV = Train(TGV_, 'TGV', "Märklin my world 29304", -1, (None,)*15, 0., functions=(), img_path="MW_TGV.png", trips_contacts=False)
MW_LOK = Train(GUETER, 'Dampf/Diesel/E', 'Märklin my world 36270', -1, (None,)*15, 0., functions=(), img_path="MW_Lok.png", trips_contacts=False)

TRAINS = [ICE, S, BUS, E_RB, E_BW, ROT, BEIGE,   E40, DIESEL, DAMPF]
EXT_TRAINS = [MW_TGV, MW_LOK]

TRAINS_BY_NAME = {train.id: train for train in TRAINS + EXT_TRAINS}


def obstacle(no=None):
    no = random.randint(0, 1_000_000_000) if no is None else no
    return Train(TrainInfo("Gesperrt", "🚧"), "Gesperrt", str(no), -1, (0,)*15, 0., img_path="Baustelle.png", functions=())


def train_by_name(name):
    if name.startswith("Gesperrt"):
        return obstacle(no=int(name[11:]))
    return TRAINS_BY_NAME[name]