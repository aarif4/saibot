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
        self.HEADLESS = True # whether to show the simplified map or not
        self.scouting_candidates = None # will hold a list of sc2.position.Point2
        self.scout_target_idx = None # will hold index of the scouting_candidate we want to go to 
        self.scout_target_loc = None # will hold the sc2.position.Point2 that the scout is currently heading towards
        self.scout_id = None #TODO: Use the scout ID to tell if they've died or not
        self.scout_num_deaths = 0 # depending on how often a scout dies, we may want to increase our radius from the enemy
        self.scout_saturated_iteration = None
        self.scout_saturated_candidates = None

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
    
    def sort_shortest_bitonic_tour(self):
        # adapted from here: https://stackoverflow.com/questions/25552128/find-shortest-path-through-points-in-2d-plane
        # basic idea:
        # (Note that d[1,2] = dist(p1,p2))
        # d[1,3] = d[1,2] + dist(p2,p3).
        # d[i,j] = d[i,j-1] + dist(j-1,j) for i < j-1.
        #
        # Algorithmically:
        # for k in range(j-2)
        # d[j-1,j] = min( d[k,j-1] + dist(k,j) ) for 1 <= k < j-1
        pass

    def get_viable_scouting_candidates(self, min_dist_to_target):
        # let's take inventory on where our supporting units are w.r.t the scouting candidate locations
        # basically for each unknown location, find the closest unit/structure and see if the distance is less than min_dist_to_target
        # we'll create a vector of the same length as self.scouting_candidates and will be toggled 0 if not a viable candidate location and 1 if it still is    
        scouting_candidates_viability = np.ones(len(self.scouting_candidates))
        for i in range(len(self.scouting_candidates)):
            candidate = self.scouting_candidates[i]
            # find closest unit's distance to candidate
            unit_dist = self.units.closest_distance_to(candidate)
            # find closest structure's distance to candidate
            struct_dist = self.structures.closest_distance_to(candidate)
            # this candidate location is still viable to scout IFF no structure or unit are "close enough" to it
            scouting_candidates_viability[i] = struct_dist > min_dist_to_target and unit_dist > min_dist_to_target
        
        # return the indexes of the candidates that still needs to be scouted (can be empty!)
        return list(np.where(scouting_candidates_viability == 1)[:][0])


    def do_frantic_search(self):
        # in 30 iterations, begin randomizing list of candidates and force to search every one of them
        num_iterations_to_wait = 30

        # if first time, set it equal to the current iteration value
        if self.scout_saturated_iteration is None:
            self.scout_saturated_iteration = self.iteration
        
        move_to = self.start_location
        if self.iteration >= self.scout_saturated_iteration + num_iterations_to_wait:
            # then it's time to randomize and go one by one
            if self.scout_saturated_candidates is None or len(self.scout_saturated_candidates) == 0:
                self.scout_saturated_candidates = self.scouting_candidates
                random.shuffle(self.scout_saturated_candidates) # this will random reorganize the list
            # choose where to go
            move_to = self.scout_saturated_candidates[0]
            self.scout_saturated_candidates.pop(0)

        return move_to

    def get_next_viable_scouting_candidate(self, viable_candidates):
        self.scout_saturated_iteration = None # reset saturation counter

        if all([vc < self.scout_target_idx for vc in viable_candidates]):
            # means every viable candidate is an index lower than the current target candidate
            # use the FIRST viable candidate in the list instead and break
            self.scout_target_idx = viable_candidates[0]
        else:
            for vc in viable_candidates:
                # we want to go to the NEXT candidate in the list that's at or past the current self.scout_target_idx's value/idx
                if vc >= self.scout_target_idx:
                    self.scout_target_idx = vc
                    break # break out since we found the first occurance past our current target
        
        return self.scouting_candidates[self.scout_target_idx % len(self.scouting_candidates)]

    async def scout(self):
        # 1. if there's no scout, make a scout
        # 2. if you have a scout, go to the following places in this priority
        #    a. the enemy's start location
        #    b. an unknown location
        #       i. if the scout has reached an unknown location, go to the next unknown location
        #       ii. if the scout has been to every unknown location, then start looking around again
        scout_unitid = sc2.constants.OBSERVER
        scout_bldg_unitid = sc2.constants.ROBOTICSFACILITY
        min_dist_to_target = 10 # + self.scout_num_deaths # this is the min dist the scout needs to be before it can move on to the next point
        max_dist_to_target = 30
        dist_to_target = min_dist_to_target + self.scout_num_deaths
        if dist_to_target > max_dist_to_target:
            dist_to_target = max_dist_to_target # cap the maximum radius

        # if this is empty make a list containing the enemy's starting location as well as other hidden locations
        if self.scouting_candidates is None:
            self.scouting_candidates = self.enemy_start_locations + list(self.expansion_locations_list)#list(self.game_info.vision_blockers)        
        
        # if this is the first time, set scout_target_idx to 0 so that it starts with the first candidate
        if self.scout_target_idx is None:
            self.scout_target_idx = 0
        
        # if first time, set scout_target_loc to a sc2.position.Point2 value
        if self.scout_target_loc is None:
            self.scout_target_loc = self.scouting_candidates[self.scout_target_idx % len(self.scouting_candidates)]
        
        if len(self.units(scout_unitid)) > 0:
            # get the scout that's alive
            scout = self.units(scout_unitid)[0]

            # if first time, save the scout's id
            if self.scout_id is None:
                self.scout_id = scout.tag
            # if this a new scout spawn, increment the death count and update self.scout_id
            if self.scout_id != scout.tag:
                self.scout_num_deaths = self.scout_num_deaths + 1
                self.scout_id = scout.tag

            # get the indices of the candidates that still need to be explored
            viable_candidates = self.get_viable_scouting_candidates(dist_to_target)
            
            if len(viable_candidates) == 0:
                # means we have units/structures on every scouting candidate location
                # go home and wait for a bit.
                # if things are too peaceful, let's wait and then:
                # 1. randomize list of scouting candidates
                # 2. go over every scouting candidate
                # 3. repeat 1. when you've reached the last candidate
                move_to = self.do_frantic_search()
                if move_to == self.scout_target_loc and not scout.is_idle:
                    # scout is already moving to that target, ignore it
                    pass
                else:
                    if move_to == self.start_location:
                        print('SATURATED STATE | Sending scout back to home base while we wait 30 iterations :', move_to)
                    else:
                        print('SATURATED STATE | Sending scout to the following random location :', move_to)
                    self.scout_target_loc = move_to
                    scout.move(move_to)
                
            elif len(viable_candidates) != 0 and scout.is_idle:
                # means the scout needs to go to the next viable scouting candidate
                # based on the list of viable scouting candidates, find one that has an idx greater than the current target candidate's idx
                # - if there is one, go to that
                # - else, loop back to the start and go to the first viable scouting candidate
                move_to = self.get_next_viable_scouting_candidate(viable_candidates)
                print('   IDLE STATE   | Sending scout to location', self.scout_target_idx,'out of',len(self.scouting_candidates),':', move_to)
                self.scout_target_loc = move_to
                scout.move(move_to)

            elif self.scout_target_loc is not None and scout.distance_to(self.scout_target_loc) < dist_to_target:
                # means the scout is moving and has reached the threshold to the target candidate location
                # time to pivot before they get blasted
                move_to = self.get_next_viable_scouting_candidate(viable_candidates)
                print('  MOVING STATE  | Sending scout to location', self.scout_target_idx,'out of',len(self.scouting_candidates),':', move_to)
                self.scout_target_loc = move_to
                scout.move(move_to)
        else:
            # train a scout first
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
