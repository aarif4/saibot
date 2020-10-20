import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from core.terran import Terran
from core.protoss import Protoss
from core.zerg import Zerg

race = 2
realtime = False
difficulty=Difficulty.Easy
if race == 1:
    run_game(maps.get("Abyssal Reef LE"), [
        Bot(Terran().get_race(), Terran()),
        Computer(Race.Protoss, difficulty)
], realtime=realtime)
elif race == 2:
    run_game(maps.get("Abyssal Reef LE"), [
        Bot(Protoss().get_race(), Protoss()),
        Computer(Race.Protoss, difficulty)
    ], realtime=realtime)
elif race == 3:
    run_game(maps.get("Abyssal Reef LE"), [
        Bot(Zerg().get_race(), Zerg()),
        Computer(Race.Protoss, difficulty)
    ], realtime=realtime)
