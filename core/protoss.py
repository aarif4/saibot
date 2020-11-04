import sc2
import random
import cv2 # pip install opencv-python
import numpy as np
import time
import os


class Protoss(sc2.BotAI):
    @staticmethod
    def get_race():
        return sc2.Race.Protoss
    

    def __init__(self):
        self.iterations_per_minute = 165
        self.max_workers = 65
        self.do_something_after = 0
        self.train_data = []
        self.HEADLESS = False # whether to show the simplified map or not

    def on_end(self, game_result):
        print('---on_end called---')
        print(game_result)

        if game_result == sc2.Result.Victory:
            if not os.path.isdir('train_data'):
                os.mkdir('train_data')
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration: int):
        self.iteration = iteration
        #print(self.iteration/self.iterations_per_minute)
        await self.scout()
        await self.intel()
        await self.distribute_workers()
        await self.increase_supply_cap()
        await self.build_worker_units()
        await self.build_vespene_gas_structure()
        await self.build_townhall_structure()
        await self.build_combat_structures()
        await self.train_combat_units()
        await self.attack_enemy()


    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0],3), np.uint8)

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)
        
        townhall_unitid = sc2.constants.NEXUS
        worker_unitid   = sc2.constants.PROBE
        supply_unitid = sc2.constants.PYLON
        vgs_unitid = sc2.constants.ASSIMILATOR
        combat_bldg_unitid = sc2.constants.GATEWAY
        combat_bldg_addon_unitid = sc2.constants.CYBERNETICSCORE
        combat_bldg_addon2_unitid = sc2.constants.STARGATE
        arial_combat_unitid = sc2.constants.VOIDRAY
        scout_unitid = sc2.constants.OBSERVER

        # UNIT/STRUCT: [SIZE,(BGR COLOR)]
        draw_dict = {
            townhall_unitid:            [15, (0, 255, 0),1],
            supply_unitid:              [3, (20, 235, 0),1],
            worker_unitid:              [1, (55, 200, 0),0],
            vgs_unitid:                 [2, (0, 200, 55),1],
            combat_bldg_unitid:         [3, (200, 100, 0),1],
            combat_bldg_addon_unitid:   [3, (150, 150, 0),1],
            combat_bldg_addon2_unitid:  [5, (255, 0, 0),1],
            arial_combat_unitid:        [3, (255, 100, 0),0],
            scout_unitid:               [1, (255,255,255),0]
        }
        for unit_type in draw_dict:
            if draw_dict[unit_type][-1]:
                # it's a struct
                for unit in self.structures(unit_type).ready:
                    pos = unit.position
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1],-1)
            else:
                for unit in self.units(unit_type).ready:
                    pos = unit.position
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1],-1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(arial_combat_unitid)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0


        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data,0)
        
        if not self.HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2,fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)
    

    async def scout(self):
        scout_unitid = sc2.constants.OBSERVER
        scout_bldg_unitid = sc2.constants.ROBOTICSFACILITY

        if len(self.units(scout_unitid)) > 0:
            scout = self.units(scout_unitid)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.ranodm_location_variance(enemy_location)
                scout.move(move_to)
        else:
            for rf in self.structures(scout_bldg_unitid).ready:
                if rf.is_idle:
                    if self.can_afford(scout_unitid) and self.supply_left > 0:
                        rf.train(scout_unitid)


    def ranodm_location_variance(self,enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20,20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20,20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = sc2.position.Point2(sc2.position.Pointlike((x,y)))
        return go_to
        
    async def build_worker_units(self):
        townhall_unitid = sc2.constants.NEXUS
        worker_unitid   = sc2.constants.PROBE
        
        if len(self.structures(townhall_unitid))*16 > len(self.units(worker_unitid)) and len(self.units(worker_unitid)) < self.max_workers:
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
        
        if self.structures(townhall_unitid).amount < ((self.iteration / self.iterations_per_minute) / 2) and self.can_afford(townhall_unitid) and not self.already_pending(townhall_unitid):
            await self.expand_now()


    async def build_combat_structures(self):
        supply_unitid = sc2.constants.PYLON
        combat_bldg_unitid = sc2.constants.GATEWAY
        combat_bldg_addon1_unitid = sc2.constants.CYBERNETICSCORE
        combat_bldg_addon2_unitid = sc2.constants.STARGATE
        combat_bldg_addon3_unitid = sc2.constants.ROBOTICSFACILITY
        

        if self.structures(supply_unitid).ready.exists:
            target_supply = self.structures(supply_unitid).ready.random
            if self.structures(combat_bldg_unitid).ready.exists and not self.structures(combat_bldg_addon1_unitid):
                if self.can_afford(combat_bldg_addon1_unitid) and not self.already_pending(combat_bldg_addon1_unitid):
                    await self.build(combat_bldg_addon1_unitid, near=target_supply)
            elif len(self.structures(combat_bldg_unitid)) < 1:
                if self.can_afford(combat_bldg_unitid) and not self.already_pending(combat_bldg_unitid):
                    await self.build(combat_bldg_unitid, near=target_supply)
            if self.structures(combat_bldg_addon1_unitid).ready.exists:
                if len(self.structures(combat_bldg_addon2_unitid)) < ((self.iteration / self.iterations_per_minute)):
                    if self.can_afford(combat_bldg_addon2_unitid) and not self.already_pending(combat_bldg_addon2_unitid):
                        await self.build(combat_bldg_addon2_unitid, near=target_supply)
            if self.structures(combat_bldg_addon1_unitid).ready.exists:
                if len(self.structures(combat_bldg_addon3_unitid)) < 1:
                    if self.can_afford(combat_bldg_addon3_unitid) and not self.already_pending(combat_bldg_addon3_unitid):
                        await self.build(combat_bldg_addon3_unitid, near=target_supply)


    async def train_combat_units(self):
        combat_bldg_unitid = sc2.constants.GATEWAY
        combat_bldg_addon_unitid = sc2.constants.CYBERNETICSCORE
        combat_bldg_addon2_unitid = sc2.constants.STARGATE
        basic_combat_unit = sc2.constants.STALKER
        arial_combat_unit = sc2.constants.VOIDRAY
        
        for sg in self.structures(combat_bldg_addon2_unitid).ready:
            if sg.is_idle:
                if self.can_afford(arial_combat_unit) and self.supply_left > 0:
                    sg.train(arial_combat_unit)


    def find_target(self, state):
        if len(self.enemy_units) > 0:
            return random.choice(self.enemy_units)
        elif len(self.enemy_structures) > 0:
            return random.choice(self.enemy_structures)
        else:
            return self.enemy_start_locations[0] # since always 1
        
    async def attack_enemy(self):
        townhall_unitid = sc2.constants.NEXUS
        basic_combat_unit = sc2.constants.STALKER
        arial_combat_unit = sc2.constants.VOIDRAY

        if len(self.units(arial_combat_unit).idle) > 0:
            choice = random.randrange(0,4)
            target = False
            if self.iteration > self.do_something_after:
                if choice == 0:
                    #no attack
                    wait = random.randrange(20,165)
                    self.do_something_after = self.iteration + wait
                elif choice == 1:
                    if len(self.enemy_units) > 0:
                        target = self.enemy_units.closest_to(random.choice(self.structures(townhall_unitid)))
                elif choice == 2:
                    if len(self.enemy_structures) > 0:
                        target = random.choice(self.enemy_structures)
                elif choice == 3:
                    target = self.enemy_start_locations[0]
                
                if target:
                    for vr in self.units(arial_combat_unit).idle:
                        vr.attack(target)
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y,self.flipped])
