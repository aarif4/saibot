import sc2

class Terran(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Terran
    
    async def on_step(self, iteration: int):
        await self.distribute_workers()
        await self.increase_supply_cap()
        await self.build_worker_units()


    async def build_worker_units(self):
        townhall_unitid = sc2.constants.COMMANDCENTER
        worker_unitid   = sc2.constants.SCV
        
        for townhall in self.structures(townhall_unitid):
            if townhall.is_ready and townhall.is_idle:
                if self.can_afford(worker_unitid):
                    townhall.train(worker_unitid)


    async def increase_supply_cap(self):
        MIN_SUPPLY_THRESH = 5
        townhall_unitid = sc2.constants.COMMANDCENTER
        supply_unitid = sc2.constants.SUPPLYDEPOT
        
        if self.supply_left < MIN_SUPPLY_THRESH and not self.already_pending(supply_unitid):
            for townhall in self.structures(townhall_unitid).ready:
                if self.can_afford(supply_unitid):
                    await self.build(supply_unitid, near=townhall)