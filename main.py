import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from core.zerg import Zerg



run_game(maps.get("Abyssal Reef LE"), [
    Bot(Zerg().get_race(), Zerg()),
    Computer(Race.Protoss, Difficulty.Medium)
], realtime=True)
