import sc2

class Zerg(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Zerg
    
    async def on_step(self, iteration: int):
        await self.distribute_workers()
        await self.increase_supply_cap()
        await self.build_worker_units()
        await self.build_vespene_gas_structure()
        await self.build_townhall_structure()

    async def build_worker_units(self):
        townhall_unitid = sc2.constants.HATCHERY
        worker_unitid   = sc2.constants.DRONE
        
        for townhall in self.structures(townhall_unitid):
            for target_larva in self.larva.closest_n_units(townhall.position, 1):
                if self.can_afford(worker_unitid):
                    target_larva.train(worker_unitid)


    async def increase_supply_cap(self):
        MIN_SUPPLY_THRESH = 5
        townhall_unitid = sc2.constants.HATCHERY
        supply_unitid = sc2.constants.OVERLORD
        
        if self.supply_left < MIN_SUPPLY_THRESH and not self.already_pending(supply_unitid):
            for townhall in self.structures(townhall_unitid).ready:
                for target_larva in self.larva.closest_n_units(townhall.position, 1):
                    if self.can_afford(supply_unitid):
                        target_larva.train(supply_unitid)


    async def build_vespene_gas_structure(self):
        townhall_unitid = sc2.constants.HATCHERY
        vgs_unitid = sc2.constants.EXTRACTOR
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
        townhall_unitid = sc2.constants.HATCHERY
        
        if self.structures(townhall_unitid).amount < MAX_NO_TOWNHALLS and self.can_afford(townhall_unitid) and not self.already_pending(townhall_unitid):
            await self.expand_now()