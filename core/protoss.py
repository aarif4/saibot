import sc2
import random


class Protoss(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Protoss
    

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        await self.increase_supply_cap()
        await self.build_worker_units()
        await self.build_vespene_gas_structure()
        await self.build_townhall_structure()

        
    async def build_worker_units(self):
        townhall_unitid = sc2.constants.NEXUS
        worker_unitid   = sc2.constants.PROBE
        
        for townhall in self.structures(townhall_unitid):
            if townhall.is_ready and townhall.is_idle:
                if self.can_afford(worker_unitid):
                    townhall.train(worker_unitid)


    async def increase_supply_cap(self):
        MIN_SUPPLY_THRESH = 5
        townhall_unitid = sc2.constants.NEXUS
        supply_unitid = sc2.constants.PYLON
        
        if self.supply_left < MIN_SUPPLY_THRESH and not self.already_pending(supply_unitid):
            for townhall in self.structures(townhall_unitid).ready:
                if self.can_afford(supply_unitid):
                    await self.build(supply_unitid, near=townhall)


    async def build_vespene_gas_structure(self):
        townhall_unitid = sc2.constants.NEXUS
        vgs_unitid = sc2.constants.ASSIMILATOR
        min_dist = 15.0
        
        for townhall in self.structures(townhall_unitid).ready:
            vg_geysers = self.vespene_geyser.closer_than(min_dist, townhall)
            for vg_geyser in vg_geysers:
                if not self.can_afford(vgs_unitid):
                    break
                
                worker = self.select_build_worker(vg_geyser.position)
                if worker is None:
                    break
                
                if not self.structures(vgs_unitid).closer_than(1, vg_geyser).exists and not self.already_pending(vgs_unitid):
                    worker.build(vgs_unitid, vg_geyser)


    async def build_townhall_structure(self):
        MAX_NO_TOWNHALLS = 3
        townhall_unitid = sc2.constants.NEXUS
        
        if self.structures(townhall_unitid).amount < MAX_NO_TOWNHALLS and self.can_afford(townhall_unitid) and not self.already_pending(townhall_unitid):
            await self.expand_now()

