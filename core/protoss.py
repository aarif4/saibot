import sc2
from sc2.constants import  NEXUS, PROBE, PYLON, ASSIMILATOR

class Protoss(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Protoss
    

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.build_nexuses()
        #await self.rush_at_enemy(iteration)


    async def build_workers(self):
        for nexus in self.structures(NEXUS):
            if nexus.is_ready and nexus.is_idle:
                if self.can_afford(PROBE):
                    nexus.train(PROBE)


    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            for nexus in self.structures(NEXUS).ready:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexus)


    async def build_assimilators(self):
        for nexus in self.structures(NEXUS).ready:
            vaspenes = self.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.structures(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    worker.build(ASSIMILATOR, vaspene)
    
    async def build_nexuses(self):
        if self.structures(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    
    async def rush_at_enemy(self, iteration):
        if iteration == 0:
            for worker in self.workers:
                worker.attack(self.enemy_start_locations[0])

