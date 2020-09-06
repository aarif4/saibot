import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from core.zerg import Zerg
from core.protoss import Protoss



run_game(maps.get("Abyssal Reef LE"), [
    Bot(Protoss().get_race(), Protoss()),
    Computer(Race.Protoss, Difficulty.Easy)
], realtime=True)
