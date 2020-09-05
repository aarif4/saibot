import sc2

class Protoss(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Protoss
    
    async def on_step(self, iteration: int):
        await self.rush_at_enemy(iteration)
        
    async def rush_at_enemy(self, iteration):
        if iteration == 0:
            for worker in self.workers:
                worker.attack(self.enemy_start_locations[0])
