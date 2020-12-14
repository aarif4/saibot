import sc2
import random
import cv2 # pip install opencv-python
import numpy as np
import time
import os
import math
import utils # from main project
import tensorflow as tf # get keras like so: tf.keras


class Protoss(sc2.BotAI):
    """
    BotAI class to handle the Protoss race
    """
    @staticmethod
    def get_race():
        """Get the bot's race

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            {sc2.Race} -- Race that the bot's meant to play with

        Attributes Affected:
            N/A

        Attributes Referenced:
            N/A
        """
        return sc2.Race.Protoss

    def __init__(self, user_data, name):
        """Initializes class attributes using user-defined config data

        Argument Keywords:
            user_data {utils.config_data} --  has user's requested
                                              configuration settings
            name      {string}            --  has bot's name, usually is either
                                              "player_bot" or "enemy_bot"

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            logger                  {logging}   --  logging object we use
                                                    throughout the lifetime of
                                                    this class
            sim_time_min            {float}     --  current game's timekeeping
                                                    in minutes
            collect_data            {dict}      --  holds info on training data
                                                    collects for this game
            max_workers             {int}       --  maximum no. of worker units
            supply_threshold        {int}       --  min number of available
                                                    spots in army before we
                                                    need to build another
                                                    supply building
            vgs_max_radius          {int}       --  radius to search around
                                                    a townhall for vespene
                                                    geysers
            townhall_build_rate     {float}     --  we shall wait this many
                                                    minutes before we build
                                                    another townhall
            plot_map_intel          {bool}      --  flag that toggles intel
                                                    plotting
            stay_idle_until_min     {float}     --  avoid engaging the enemy
                                                    until this point in time
            model                   {dict}      --  holds info on the model
                                                    being used in this bot,
                                                    if applicable
            scout                   {dict}      --  holds info on scout's
                                                    behavior and state
            unitid                  {dict}      --  holds unitid values for
                                                    each unit/building used in
                                                    this bot

        Attributes Referenced:
            logger                  {logging}
            sim_time_min            {float}
            collect_data            {dict}
            max_workers             {int}
            supply_threshold        {int}
            vgs_max_radius          {int}
            townhall_build_rate     {float}
            plot_map_intel          {bool}
            stay_idle_until_min     {bool}
            default_nan_point2      {Point2}
            pre_pending_bldgs       {(dict(Point2,int)}
            model                   {dict}
            scout                   {dict}
            unitid                  {dict}
            dependencies            {dict}
            color_scheme            {dict}
        """
        sc2.BotAI.__init__(self)
        #----------------------------------------------------------------------
        # create a logger object so that you can use it in the bot
        #----------------------------------------------------------------------
        self.logger, self.ch = utils.build_logger(name, 'Alert')

        #----------------------------------------------------------------------
        # this declares the time elapsed in the game for this bot (in min)
        #----------------------------------------------------------------------
        self.sim_time_min = 0

        #----------------------------------------------------------------------
        # this attribute holds all the data related to training data collected
        # in this trial
        #----------------------------------------------------------------------
        self.collect_data = {
            'exists': user_data.cfg[name]['save_training_data'],
            'path': user_data.cfg[name]['training_data_dir'],
            'current_intel': None,
            'training_data': []
        }

        #----------------------------------------------------------------------
        # Some additional attributes to help during several actions:
        # - building more workers
        # - plot intel
        # - disengage from enemy for a period of time
        #----------------------------------------------------------------------
        self.bot_mode = user_data.cfg[name]['mode']
        self.logger.info('Initializing bot to run in the following mode: %s', str(self.bot_mode))
        # this defines the limit on the max number of workers this bot can make
        self.max_workers = user_data.cfg[name]['max_num_workers']
        # set the threshold before we need to create another supply bldg
        self.supply_threshold = 3
        # set the min distance a vespene geyser has to be from a townhall
        self.vgs_max_radius = 15
        # set the build rate for townhalls
        self.townhall_build_rate = 3 # build one every self.townhall_build_rate minutes
        self.max_townhalls = 3
        # this decides whether to display map intel to the user or not
        self.plot_map_intel = user_data.cfg[name]['plot_map_intel']
        # used to declare to wait for a randomized period of time [1,5) min
        # until the bot decided on the next move
        self.stay_idle_until_min = 0
        # default NaN Point2
        self.default_nan_point2 = sc2.position.Point2((-100,-100))
        # keep track of the buildings locations that are pre-pending
        self.pre_pending_bldgs = {}
        # keep track of pre-pending combat buildings
        self.pending_combat_bldgs ={}
        self.wait_pending_bldg_min = 0.5 # wait this many mins before you think about building smthg
        self.combat_bldg_build_rate = 1 # build 1 bldg every self.combat_bldg_build_rate minutes
        self.prev_target = {
            "found": False,
            "loc": None,
            "choice": np.zeros(4)
        }

        #----------------------------------------------------------------------
        # this holds all the info on the model we've decided to use
        # if applicable, then this model's downloaded in the on_start() method
        #----------------------------------------------------------------------
        self.model = {
            'exists': user_data.cfg[name]['mode'] == utils.BOT_MODE.DNN,
            'path': user_data.cfg[name]['model_location'],
            'model': None
        }

        #----------------------------------------------------------------------
        # this set of attributes are used for scouting
        # TODO: it would be nice to remove "target_candidate_idx" and instead
        # just "cycle" through the ring of explorable sites using the list we
        # can get from self.expansion_locations_list
        #----------------------------------------------------------------------
        self.scout = {
            "orig_unitid": sc2.constants.OBSERVER,
            "use_worker": user_data.cfg[name]['use_worker_scout'],
            "tag": -1, # holds the scout unit's tag, which is an integer value
            "num_deaths": 0, # keep track
            "candidate_sites": [], # [sc2.position.Point2]
            "target_candidate_idx": -1, # index of "candidate_sites"
            "target_candidate_loc": self.default_nan_point2, # invalid location on map
            "saturated_time_min": 0, # time at which we've exhausted all sites
            "saturation_timeout_min": 2, # amount of time to wait while in saturation state
            "saturated_candidate_sites": [], # [sc2.position.Point2] | list of sites to re-check
            "min_distance": 10, # min distance for scout to be at target before it's a success
            "max_distance": 30 # max distance to be from target before it's a success
        }
        if self.scout['use_worker']:
            self.logger.info('Going to use a worker unit as a scout!')
        else:
            self.logger.info('Going to use a proper scouting unit as a scout')

        #----------------------------------------------------------------------
        # categorize the list of unitids that we use in this race
        #----------------------------------------------------------------------
        self.unitid = {
            "townhall_bldg": sc2.constants.NEXUS,
            "worker": sc2.constants.PROBE,
            "supply_bldg": sc2.constants.PYLON,
            "vgs_bldg": sc2.constants.ASSIMILATOR,
            "combat_bldg": sc2.constants.STARGATE,
            "combat_bldg_addons": [sc2.constants.GATEWAY, sc2.constants.CYBERNETICSCORE],
            "combat": sc2.constants.VOIDRAY,
            "scout": sc2.constants.PROBE if self.scout["use_worker"] else sc2.constants.OBSERVER,
            "scout_bldg": sc2.constants.NEXUS if self.scout["use_worker"] else
                sc2.constants.ROBOTICSFACILITY
        }

        #----------------------------------------------------------------------
        # categorize the list of unitids that we use in this race
        # - building dependencies can be a list
        # - unit dependencies are singular
        #----------------------------------------------------------------------
        self.dependencies = {
            sc2.constants.NEXUS: [],
            sc2.constants.PYLON: [],
            sc2.constants.ASSIMILATOR: [],
            sc2.constants.GATEWAY: [],
            sc2.constants.CYBERNETICSCORE: [sc2.constants.GATEWAY],
            sc2.constants.STARGATE: [sc2.constants.CYBERNETICSCORE],
            sc2.constants.ROBOTICSFACILITY: [sc2.constants.CYBERNETICSCORE],
            sc2.constants.VOIDRAY: sc2.constants.STARGATE
        }

        #----------------------------------------------------------------------
        # color scheme (BGR) to use in intel for each unitid
        #----------------------------------------------------------------------
        self.color_scheme = {
            self.unitid["townhall_bldg"]: (0, 255, 0),
            self.unitid["supply_bldg"]: (20, 235, 0),
            self.unitid["vgs_bldg"]: (0, 200, 55),
            self.unitid["worker"]: (55, 200, 0)
        }
        # TODO: Remove unused globals
        # If we're dealing with an enemy unit, make sure we're not retargeting the same one, if so,
        # then tell them to keep doing what they're doing....? So maybe keep track of tag and if we
        # ever choose a diff decision, reset tag
        #

        # removing this since this set of buildings is not important enough to make a decision with
        # self.color_scheme.update(\
        #     dict(\
        #         zip(\
        #             self.unitid["combat_bldg_addons"],\
        #             [(150, 150, 0), (200, 100, 0)])))
        self.color_scheme.update({
            self.unitid["combat_bldg"]: (255, 0, 0),
            self.unitid["combat"]: (255, 100, 0),
            self.scout['orig_unitid']: (255,255,255),
            "enemy_townhall": (0, 0, 255),
            "enemy_structure": (200, 50, 212),
            "enemy_worker": (55, 0, 155),
            "enemy_combat": (50, 0, 215)
        })

    async def on_start(self):
        """Function called in the beginning of a bot's lifecycle. If this bot's
        meant to apply a DNN model, then that model's imported in this method

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            sc2.Race

        Attributes Affected:
            model   {dict} -- if the bot's meant to use a model, we import it
                              in this function using tf.keras

        Attributes Referenced:
            model   {dict}      --  "exists"
                                    "model"
                                    "path"
            logger  {logging}
        """
        if self.model['exists']:
            self.logger.info('Loading the following model: %s', self.model['path'])
            self.model['model'] = tf.keras.models.load_model(self.model['path'])

    async def on_end(self, game_result):
        """Function called in the end of a bot's lifecycle. Logs the final
        result of the trial into an .npy file, if requested by the user.

        Argument Keywords:
            game_result {sc2.Result} -- Is the final result of the current game

        Raises:
            N/A

        Returns:
            sc2.Race

        Attributes Affected:
            N/A

        Attributes Referenced:
            logger          {logging}
            collect_data    {dict}      --  "exists"
                                            "path"
                                            "training_data"
        """
        self.logger.info('Ended the game: %s', game_result.name)
        if game_result == sc2.Result.Victory:
            # save training data if we need some
            if self.collect_data['exists']:
                np.save(
                    "{}/{}.npy".format(
                        self.collect_data['path'],
                        str(int(time.time()))),
                    np.array(self.collect_data['training_data']))

        # delete your logger because it'll persist
        self.logger.info('Closing bot...')
        self.logger.removeHandler(self.ch)
        del self.logger, self.ch

    async def on_step(self, iteration: int):
        """Function called at each iteration of the bot's lifecycle. This
        function then does several things:
        - track the time in the bot (doesn't matter if it's not realtime)
        - gather intel on the state of the game (enemies vs. allied forces)
        - distribute workers to gather resources
        - start gathering enough resources to send a scout
        - if we've reached our threshold of units, increase the supply cap
        - build worker units that'll gather resources
        - build vespene gas structures to gather different kinds of resources
        - build a townhall structure if enough time has passed
        - build combat structures if we have enough resources
        - decide on how to engage the enemy

        Argument Keywords:
            iteration {int} -- Is the "tick" value of the timekeeping component
                               (This is unused since it isn't human-readable)

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            sim_time_min {float} -- current time in the game (in minutes)

        Attributes Referenced:
            sim_time_min {float}
        """
        #----------------------------------------------------------------------
        # increment our timekeeper
        #----------------------------------------------------------------------
        self.sim_time_min = (self.state.game_loop/22.4)/60

        #----------------------------------------------------------------------
        # run all of the bot's actions
        #----------------------------------------------------------------------
        await self.gather_intelligence()
        await self.distribute_workers() # only affects "idle" workers
        await self.scout_enemy()
        await self.build_supply_cap()
        await self.train_worker_units()
        await self.build_vespene_gas_structure()
        await self.build_townhall_structure()
        await self.build_combat_structures()
        await self.train_combat_units()
        await self.engage_enemy()

    async def gather_intelligence(self):
        """This function helps gather information on the state of the armies of
        both our forces and our enemy's forces. We'll draw circles for each
        type of unit as well as the amount of resources that we've collected

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            self.collect_data {dict} -- modify 'current_intel' key to what is
                                        generated by the end of this function

        Attributes Referenced:
            color_scheme    {dict}
            unitid          {dict}  --  "townhall_bldg"
                                        "supply_bldg"
                                        "worker"
                                        "scout"
                                        "combat"
            collect_data    {dict}  --  "current_intel"
        """
        use_radius = True
        radius_scale = 7

        #----------------------------------------------------------------------
        # make an empty grid that fits the game's map's size
        #----------------------------------------------------------------------
        game_data = np.zeros(
            (self.game_info.map_size[1], self.game_info.map_size[0], 3),
            np.uint8
        )

        #----------------------------------------------------------------------
        # start "coloring-in" your own structures and units as well as your
        # enemy's structures and units
        #----------------------------------------------------------------------
        for key in list(self.color_scheme.keys()):
            if not isinstance(key, sc2.UnitTypeId):
                continue # don't plot keys that are not sc2.constant types (those are enemy's stuff)

            for struct in self.structures(key):
                pos = struct.position
                # get the radius
                if struct.type_id == self.unitid["townhall_bldg"]:
                    radius = struct.footprint_radius*radius_scale if use_radius else 15
                elif struct.type_id == self.unitid["supply_bldg"]:
                    radius = struct.footprint_radius*radius_scale if use_radius else 3
                else:
                    radius = struct.footprint_radius*radius_scale if use_radius else 5
                # get the color
                color = self.color_scheme[struct.type_id]
                # plot a circle in our map with the given radius and color
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(radius), color, -1)

        for struct in self.enemy_structures:
            pos = struct.position
            # check if it's a townhall structure
            if struct.name.lower() in ["nexus", "commandcenter", "hatchery"]:
                radius = struct.footprint_radius*radius_scale if use_radius else 15
                color = self.color_scheme["enemy_townhall"]
            else: # it's a non-townhall structure
                radius = struct.footprint_radius*radius_scale if use_radius else 5
                color = self.color_scheme["enemy_structure"]
            # plot a circle in our map with the given radius and color
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(radius), color, -1)

        for key in list(self.color_scheme.keys()):
            if not isinstance(key, sc2.UnitTypeId):
                continue # don't plot keys that are not sc2.constant types (those are enemy's stuff)

            for unit in self.units(key):
                pos = unit.position
                # get the radius
                if unit.type_id == self.unitid["worker"]:
                    radius = unit.radius*radius_scale if use_radius else 1
                else:
                    radius = unit.radius*radius_scale if use_radius else 3
                # get the color
                if unit.tag == self.scout["tag"]: # it's a scout fosho, use scout color
                    color = self.color_scheme[self.scout['orig_unitid']]
                else:
                    color = self.color_scheme[unit.type_id]
                # plot a circle in our map with the given radius and color
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(radius), color, -1)

        for unit in self.enemy_units.filter(lambda x: not x.is_cloaked):
            pos = unit.position
            # check if it's a worker unit
            if unit.name.lower() in ["probe", "scv",  "drone"]:
                radius = unit.radius*radius_scale if use_radius else 1
                color = self.color_scheme["enemy_worker"]
            else: # consider it a combat unit
                radius = unit.radius*radius_scale if use_radius else 3
                color = self.color_scheme["enemy_combat"]
            # plot a circle in our map with the given radius and color
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(radius), color, -1)

        #----------------------------------------------------------------------
        # plot some auxillary information detailing our level of various
        # resources and army units
        #----------------------------------------------------------------------
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / max(self.supply_cap,1)
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0
        if self.supply_cap == self.supply_left:
            pass#print('hi')
        military_weight = \
            len(self.units(self.unitid["combat"])) / max((self.supply_cap-self.supply_left), 1)
        if military_weight > 1.0:
            military_weight = 1.0

        # worker/supply ratio
        cv2.line(game_data, (0, 22), (int(line_max*military_weight), 22), (250, 250, 200), 3)
        # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 17), (int(line_max*plausible_supply), 17), (220, 200, 200), 3)
        # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 12), (int(line_max*population_ratio), 12), (150, 150, 150), 3)
        # gas / 1500
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)
        # minerals minerals/1500
        cv2.line(game_data, (0, 2), (int(line_max*mineral_ratio), 2), (0, 255, 25), 3)

        # save this data in self.collect_data's 'current_intel' key. Can be used for diff things:
        # 1. Save in training data
        # 2. Input to DNN Model
        # 3. Plotting intel map
        self.collect_data['current_intel'] = cv2.flip(game_data, 0)

        # now, plot the intel if requested by the user
        if self.plot_map_intel:
            resized = cv2.resize(self.collect_data['current_intel'], dsize=None, fx=2, fy=2)
            cv2.imshow('Map Intel', resized)
            cv2.waitKey(1)

    def get_viable_scouting_candidates(self, min_dist_to_target):
        """This method surveys the list of candidate scouting sites and see
        which sites do not have any of our units near them. Those sites are
        considered to be viable sites because they're considered to be
        "unexplored". We use min_dist_to_target to figure out how close do our
        units or structures need to be to a site to consider it "explored".

        We return a list of indexes relating to the list in
        self.scout["candidate_sites"]. These indexes indicate which sites are
        unexplored. If every site has been explored, an empty list will be
        returned.

        Argument Keywords:
            min_dist_to_target  {float} -- minimum distance that familiar units
                                           and/or structures must be to a site
                                           to consider it "unexplorable".

        Raises:
            N/A

        Returns:
            {list} -- list of integer list of explorable candidate sites. Can
                      be an empty list if all the sites have been explored

        Attributes Affected:
            N/A

        Attributes Referenced:
            scout   {dict}  --  "candidate_sites"
        """
        explorable_sites = np.ones(len(self.scout["candidate_sites"]))
        for i in range(len(self.scout["candidate_sites"])):
            candidate = self.scout["candidate_sites"][i]
            # find closest unit(s) to this site
            unit_dist = self.units.closest_distance_to(candidate)
            # find closest structure(s) to this site
            struct_dist = self.structures.closest_distance_to(candidate)
            # this site is considered "explorable" IFF no unit or structure are
            # "close enough" to it
            explorable_sites[i] = \
                struct_dist > min_dist_to_target and unit_dist > min_dist_to_target

        # return the indexes of the candidate sites that we should explore
        # (this list can be empty!)
        return list(np.where(explorable_sites == 1)[:][0])

    def do_frantic_search(self):
        """When we don't have any explorable sites, the sout enters a
        "saturated" state. Saturated scouts are scouts that can't seem to find
        any explorable sites to scout because friendly units and structures are
        already close enough to every one of these candidate sites. As a
        result, while we can't find explorable sites, the scout shall randomize
        the list of candidate sites and explore them one by one.

        To summarize, this method will do the following actions:
        1. Pauses all scouting actions for 2 minutes when it has just enterd
           a saturated state. During that time, the scout will return to the
           bot's main base.
           - if the scout dies, then the new scout'll go to the bot's main base
        2. Shuffle the candidate sites and go to the first one on the list.
           Once a saturated bot's been tasked to go to that site, it is removed
           from this shuffled list of candidate sites.
        3. Once the bot reaches that site, it proceeds to going to the next-
           first site in the shuffled list. As expected, that site is removed
           once the saturated bot is tasked.
        4. Task 2 & 3 repeat while the scout is still in a saturated state. If
           we reach the last site and the scout's still in a saturated state,
           then reshuffle the list of candidate sites and go through Tasks 2 &
           3 again.

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            {sc2.position.Point2} -- site that a saturated scout shall approach

        Attributes Affected:
            scout   {dict} -- "saturated_time_min" is set to the current time
                              plus 2 minutes.
                              "saturated_candidate_sites" is set to the list of
                              candidate sites, shuffled, and pop()ed out as the
                              saturated scout is tasked

        Attributes Referenced:
            scout           {dict}  -- "saturated_candidate_sites"
                                        "candidate_sites"
                                        "saturated_time_min"
                                        "saturation_timeout_min"
            sim_time_min    {float}    
        """
        if not self.scout["saturated_candidate_sites"]:
            self.scout["saturated_candidate_sites"] = self.scout["candidate_sites"]
            random.shuffle(self.scout["saturated_candidate_sites"])

        # in X minutes, begin randomizing list of candidate sites and force the
        # scout to search every one of them
        if self.scout["saturated_time_min"] == 0:
            self.scout["saturated_time_min"] = \
                self.sim_time_min + self.scout["saturation_timeout_min"]

        # by default, we'll have the saturated scout go to the bot's main base
        move_to = self.start_location

        # if the timeout has already passed, go to a site in shuffled list
        if self.sim_time_min >= self.scout["saturated_time_min"]:
            # go to the firstmost shuffled site
            move_to = self.scout["saturated_candidate_sites"][0]
            # remove that site since the scout shall be tasked
            self.scout["saturated_candidate_sites"].pop(0)

        return move_to

    def get_next_viable_scouting_candidate(self, explorable_sites):
        """From the list of explorable candidate sites, choose the next
        site in our list.

        Argument Keywords:
            explorable_sites {list} -- list of site indexes that're explorable

        Raises:
            N/A

        Returns:
            {sc2.position.Point2} -- Position of target site the scout should
                                     navigate towards

        Attributes Affected:
            scout   {dict} -- "target_candidate_idx" is the index of the site
                              that the scout's targeting right now

        Attributes Referenced:
            scout   {dict}  --  "saturated_time_min"
                                "target_candidate_idx"
                                "candidate_sites"
        """
        # reset saturation counter since the scout's not in a saturated state
        self.scout["saturated_time_min"] = 0

        # from the list of explorable sites' index, look for one that's past
        # the scout's current target site's index
        if all([i < self.scout["target_candidate_idx"] for i in explorable_sites]):
            # if we couldn't find one, then go to the FIRST explorable site
            self.scout["target_candidate_idx"] = explorable_sites[0]
        else: # there's a site to explore, find it
            for es in explorable_sites:
                # we want to go to the NEXT candidate in the list that's past
                # the current self.scout["target_candidate_idx"]'s value
                if es > self.scout["target_candidate_idx"]:
                    self.scout["target_candidate_idx"] = es
                    # we found the first occurance past our current target
                    break

        # get the target site's position
        i = self.scout["target_candidate_idx"] % len(self.scout["candidate_sites"])
        return self.scout["candidate_sites"][i]

    async def scout_enemy(self):
        """This method encapsulates the actions of a scout.
        - if there's no scout, make one
        - task the scout to go to an unexplored site
        - wait until it has reached its target site before tasking it again

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            scout {dict} -- "candidate_sites" initialized with a list of
                            expansion sites
                            "tag" modified to hold the current scout's ID
                            "num_deaths" incremented the more scouts that die
                            "target_candidate_loc" set to the new target site

        Attributes Referenced:
            scout                   {dict}  --  "min_distance"
                                                "num_deaths"
                                                "max_distance"
                                                "candidate_sites"
                                                "tag"
                                                "target_candidate_loc"
            unitid                  {dict}  --  "scout"
                                                "scout_bldg"
            default_nan_point2  {Point2}    --  default NaN point that we'll
                                                use to figure out if this is
                                                the first time running a scout
        """
        #----------------------------------------------------------------------
        # get the acceptable distance our units/structures need to be from a
        # site to be considered "explored".
        #
        # These are mainly arbitrary values I've found to work pretty well and
        # I don't think they need to be changed anytime soon
        #----------------------------------------------------------------------
        dist_to_target = \
            min(self.scout["min_distance"] + self.scout["num_deaths"], self.scout["max_distance"])

        #----------------------------------------------------------------------
        # Set up list of explorable sites, which are basically a list of areas
        # an enemy can go to expand their army
        #----------------------------------------------------------------------
        # At every fcn call:
        # 1. Get complete list
        self.scout["candidate_sites"] = self.expansion_locations_list
        # 2. Make enemy location(s) first [SAVE THIS ONE LOCALLY]
        for pos in self.enemy_start_locations:
            if pos in self.scout['candidate_sites']:
                self.scout['candidate_sites'].remove(pos)
        self.scout['candidate_sites'] = self.enemy_start_locations + self.scout['candidate_sites']      
        RAW_LIST_CANDIDATE_LOC = self.scout['candidate_sites'].copy()
        # 3. If any current townhall buildings are in there, skip those
        townhall_pos = [p.position for p in self.structures(self.unitid['townhall_bldg'])]
        rm_townhall_locs = list(set(self.scout['candidate_sites']).intersection(set(townhall_pos)))
        for loc in rm_townhall_locs:
            # if there are townhall locs in this list, remove them
            self.scout['candidate_sites'].remove(loc)
        # 4. Re-adjust target idx/loc to the right idx rel to the new list
        if self.scout['target_candidate_idx'] != -1:
            if self.scout['target_candidate_loc'] in self.scout['candidate_sites']:
                self.scout['target_candidate_idx'] = \
                            self.scout['candidate_sites'].index(self.scout['target_candidate_loc'])
            else:
                # we were going to a townhall loc, retask scout immediately
                self.scout['target_candidate_loc'] = self.default_nan_point2
                self.scout['target_candidate_idx'] = -1

        #----------------------------------------------------------------------
        # check if a scout has been trained
        #----------------------------------------------------------------------
        if not self.units(self.unitid["scout"]).exists: # check if there are scouts
            #------------------------------------------------------------------
            # train a scout if a building's available and we have enough $$$
            #------------------------------------------------------------------
            bldg = self.structures(self.unitid["scout_bldg"]).ready
            if bldg:
                bldg = bldg.random
                bldg_can_build = len(bldg.orders) < 1 and self.supply_left
                can_afford_unit = self.can_afford(self.unitid["scout"])
                has_space_in_army = self.supply_left > 0
                if bldg_can_build and can_afford_unit and has_space_in_army:
                    bldg.train(self.unitid["scout"])
                    if self.scout['tag'] == -1 and self.scout['use_worker']:
                        self.logger.debug('Training a new Worker Unit to use as a Scout Unit')
                    elif self.scout['tag'] == -1 and not self.scout['use_worker']:
                        self.logger.debug('Training the first Scout Unit')
                    elif self.scout['tag'] != -1 and self.scout['use_worker']:
                        self.logger.debug('Training another Worker Unit to use as a Scout Unit')
                    elif self.scout['tag'] != -1 and not self.scout['use_worker']:
                        self.logger.debug('Training another Scout Unit')
        else: # a scout has been trained
            #------------------------------------------------------------------
            # locate the scout you want to use
            #------------------------------------------------------------------
            if self.scout["tag"] == -1:
                # this is our first scout, choose randomly
                scout = self.units(self.unitid["scout"]).random
                # record the tag of this scout
                self.scout["tag"] = scout.tag
                if self.scout['use_worker']:
                    self.logger.debug('Using a Worker Unit to Scout (Worker#%d)', self.scout['tag'])
            else:
                # get your scout via its tag
                scout = self.units(self.unitid["scout"]).find_by_tag(self.scout["tag"])
                if not scout: # if not found, then your scout has died
                    # increase the scout death count
                    self.scout["num_deaths"] = self.scout["num_deaths"] + 1
                    # choose another one (we'll always have at least one in this else statement)
                    scout = self.units(self.unitid["scout"]).random
                    # record the tag of our scout
                    self.scout["tag"] = scout.tag
                    # let the user know
                    if self.scout['use_worker']:
                        self.logger.debug(( 'Scout Unit has died, going to use another Worker' +
                                            ' Unit (Worker#%d)'), self.scout['tag'])
                    else:
                        self.logger.debug(( 'Scout Unit has died, going to use a backup Scout Unit'+
                                            ' Unit (Scout#%d)'), self.scout['tag'])


            #------------------------------------------------------------------
            # tell the scout which explorable site it can go to
            #------------------------------------------------------------------
            viable_candidates = self.get_viable_scouting_candidates(dist_to_target)
            first_time = self.scout["target_candidate_loc"] == self.default_nan_point2
            havent_reached_target = \
                scout.distance_to(self.scout["target_candidate_loc"]) > dist_to_target
            if first_time or not havent_reached_target:
                # either this is the first time OR we're "close enough" to our target site
                # regardless, get the "next" target site
                viable_candidates = self.get_viable_scouting_candidates(dist_to_target)
                if not viable_candidates: # if you couldn't find any
                    # the scout's in a saturated state because we can't find any unexplored sites
                    # so go and frantically search for a random site to double check
                    self.scout["target_candidate_loc"] = self.do_frantic_search()
                else:
                    # we have some explorable sites to check out, choose one
                    self.scout["target_candidate_loc"] = \
                        self.get_next_viable_scouting_candidate(viable_candidates)

                # start moving to this new target site immediately
                self.logger.debug('Scout Unit is going to explore Site #%d: %s', \
                        RAW_LIST_CANDIDATE_LOC.index(self.scout['target_candidate_loc']), \
                        self.scout['target_candidate_loc'])
                scout.move(self.scout["target_candidate_loc"], queue=False)
            elif self.scout['use_worker'] and scout.is_collecting:
                # if our scout is a worker and was caught in distribute_workers(), then just force
                # its only order to be to move to the target location
                self.logger.debug(('Telling Worker/Scout Unit to stop collecting and' + \
                                    ' explore Site #%d: %s'), \
                        RAW_LIST_CANDIDATE_LOC.index(self.scout['target_candidate_loc']), \
                        self.scout['target_candidate_loc'])
                scout.move(self.scout["target_candidate_loc"], queue=False)

    async def train_worker_units(self):
        """This method tells random townhalls to train workers as long as the
        demand is still there

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            N/A

        Attributes Referenced:
            unitid  {dict}      --  "townhall_bldg"
                                    "worker"
            max_workers {int}   --  indicates the max number of workers
                                    we can make
        """
        # see if we have enough workers for each townhall
        min_num_workers_per_townhalls = \
            self.structures(self.unitid["townhall_bldg"]).amount*15 + int(self.scout['use_worker'])
        enough_workers = min_num_workers_per_townhalls <= self.units(self.unitid["worker"]).amount
        # see if the number of workers we have has hit/passed the cap
        workers_exceed_cap = self.units(self.unitid["worker"]).amount >= self.max_workers

        # if we need to create more workers, then make a new one
        if not enough_workers and not workers_exceed_cap:
            bldg = self.structures(self.unitid["townhall_bldg"]).ready
            if bldg:
                bldg = bldg.random
                bldg_can_build = len(bldg.orders) < 1 and self.supply_left
                can_afford_unit = self.can_afford(self.unitid["worker"])
                if bldg_can_build and can_afford_unit:
                    if self.units(self.unitid['worker']).amount % 10 == 0:
                        self.logger.debug(  'Training a Worker Unit, number of empty spots: %d', \
                                            self.supply_left)
                    bldg.train(self.unitid["worker"])

    async def build_supply_cap(self):
        """This method tells random townhalls to create workers as long as
        the demand is still there

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            N/A

        Attributes Referenced:
            unitid              {dict}  --  "supply_bldg"
                                            "townhall_bldg"
        """
        we_are_running_low = self.supply_left < self.supply_threshold
        a_supply_bldg_is_pending = self.already_pending(self.unitid["supply_bldg"])

        # check if we need to build a new supply building
        if we_are_running_low and not a_supply_bldg_is_pending:
            bldg = self.structures(self.unitid["townhall_bldg"]).ready
            if bldg and self.can_afford(self.unitid["supply_bldg"]):
                self.logger.debug(  "Building a Supply Building, we only have %d spots left", \
                                    self.supply_left)
                bldg = bldg.random # build near a random townhall
                await self.build(self.unitid["supply_bldg"], near=bldg)

    async def build_vespene_gas_structure(self):
        """This method builds as many vespene geyser structures as possible to
        allow the collection of vespene gas resources by workers.

        This structure is built if the following conditions are met:
        1. A townhall exists (so gas collection can be handled locally)
        2. We can afford to build a vespene geyser structure
        3. A vespene geyser structure does NOT already exist at this paritcular
        geyser location
        4. A vespene geyser structure is NOT already being built at this
        particular geyser's location

        Initially, we will check previously tasked workers and make sure that
        the requested Vespene Gas Building was built successfully. If not, then
        we'll assume that the worker is still navigating to the geyser's
        location and wait until we see a Vespene Geyser building is being built
        at that location. This also allows us to block other workers from being
        tasked to build a Vespene Geyser building at the same place.

        If the worker that we've tasked has died, then a new worker is chosen
        and tasked to go to that Vespene Geyser's location and start building
        a Vespene Gas building.

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            pre_pending_bldgs   {dict}  --  we add Point2 locations and worker
                                            tags to this dictionary to keep
                                            track of workers and Geysers that
                                            we've already taken care of. This
                                            ensures that we don't task multiple
                                            workers to build the same building
                                            on the same Vespene Geyser. 

        Attributes Referenced:
            pre_pending_bldgs   {dict}      --  both the keys & values are used
            vgs_max_radius      {float}     --  minimum radius we search for a
                                                vespene geyser from a random
                                                townhall.
            townhall_build_rate {float}     --  used to figure out how often to
                                                build a new townhall
            unitid              {dict}      --  "townhall_bldg"
                                                "vgs_bldg"
        """
        #-------------------------------------------------------------------------------------------
        # If we've already tasked certain workers to initiate the build process of some Vespene
        # Geyser buildings, let's see if they've successfully initiated that build process or not.
        #
        # If a building is pending on that location, then this worker did their job, remove this
        # Geyser from the list of pre-pending builds.
        #
        # If a building's still not pending and the worker's still alive, then they're probably
        # still making their way to the location.
        #
        # If a building is not pending and that worker died, then task a new worker to initiate a
        # build at that vespene geyser's location.
        #-------------------------------------------------------------------------------------------
        # for each pending build
        for idx in range(len(self.pre_pending_bldgs)-1, 0, -1):
            # get the location of the pending build
            pos = list(self.pre_pending_bldgs.keys())[idx]
            # get the worker that was tasked to initiate the build
            worker_tag = self.pre_pending_bldgs[pos]
            tasked_worker = self.units(self.unitid['worker']).find_by_tag(worker_tag)
            # get a list of all of the workers we've tasked to initiate builds
            tags = list(self.pre_pending_bldgs.values())

            # monitor the status of the vespene gas building initiation at this particular site
            if self.structures(self.unitid["vgs_bldg"]).closer_than(1, pos).exists:
                # If a building is pending at that Vespene Geyser's location,
                # Then pop() this location out of the list
                self.pre_pending_bldgs.pop(pos)
            elif not tasked_worker and self.can_afford(self.unitid["vgs_bldg"]):
                # If a building's not pending/ready and our worker's dead,
                # Then it's time to retask a new worker for this location

                # get the Vespene Geyser Unit() related to this location
                vg_geyser = self.vespene_geyser.closer_than(1, pos)
                if not vg_geyser:
                    self.logger.debug(('Worker#%d died trying to build Vespene Geyser Building at' +
                        '%s and we could not find that vespene geyser anymore...going to delete ' +
                        'it from pending list of builds'), worker_tag, pos)
                    self.pre_pending_bldgs.pop(pos)
                vg_geyser = vg_geyser.random # it's always going to be 1

                # get a worker that has not been tasked to initiate a build and is closest to this
                # particular vespene geyser's position
                worker = self.units(self.unitid['worker']).tags_not_in(tags)
                if worker:
                    worker = worker.closest_to(vg_geyser.position) # we should always find one
                    # if we found a valid worker, then task it to initiate the build
                    self.logger.debug(('Worker#%d died trying to build %s, retasking a new ' +
                        'Worker#%d'), worker_tag, pos, worker.tag)
                    worker.build(self.unitid["vgs_bldg"], vg_geyser)
                    self.pre_pending_bldgs[pos] = worker.tag
                else:
                    # Could not find any free workers to build a Vespene Gas Building here, skip
                    continue

        # now go over the various townhalls and see if there are any new vgs bldgs to build
        for townhall in self.structures(self.unitid["townhall_bldg"]).ready:
            # build a vespene geyser building on top of each geyser as long as we have the funds
            for vg_geyser in self.vespene_geyser.closer_than(self.vgs_max_radius, townhall):
                # find a worker to build a vgs building
                tags = list(self.pre_pending_bldgs.values())
                pos = vg_geyser.position
                worker = self.units(self.unitid['worker']).tags_not_in(tags)
                if not worker:
                    #self.logger.info('''Could not find any free workers to build a Vespene Gas 
                    #    Building at %s''', vg_geyser.position)
                    continue
                else:
                    worker = worker.closest_to(pos)
                # get a general list of vgs buildings
                available_vgs_bldgs = self.structures(self.unitid["vgs_bldg"])

                pre_pending = vg_geyser.position in self.pre_pending_bldgs.keys()
                # check that you can afford to build a vgs building
                has_rsrcs = self.can_afford(self.unitid["vgs_bldg"])
                # make sure that a vgs building does not already exists at the same spot
                bldg_done = available_vgs_bldgs.ready.closer_than(1, vg_geyser).exists
                # make sure that a vgs build is not pending
                bldg_pending = available_vgs_bldgs.not_ready.closer_than(1, vg_geyser).exists

                # Build a Vespene Geyser Building if the following is true:
                # 1. We can afford this building
                # 2. A worker exists for us to use to build a building on this geyser
                # 3. The building does NOT exist at this geyser
                # 4. This kind building is NOT currently being built at this geyser
                if not pre_pending and not bldg_done and has_rsrcs and worker and not bldg_pending:
                    self.logger.debug('Tasked Worker#%d to build a Vespene Geyser Building at %s', \
                        worker.tag, str(vg_geyser.position))
                    worker.build(self.unitid["vgs_bldg"], vg_geyser)
                    self.pre_pending_bldgs[vg_geyser.position] = worker.tag

    async def build_townhall_structure(self):
        """Builds a new townhall at a rate of 1 new building every
        self.townhall_build_rate minutes. In here, we do not suffer from
        "build initiation" issues like we do when we're building Vespene
        Geyser Buildings.

        Obviously, the number of townhalls we can build are limited due to the
        limit in available resource locations. So we'll cap the maximum number
        of Townhalls we can have to one less (because we do not want to build
        on top of an enemy's location).

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            N/A

        Attributes Referenced:
            sim_time_min        {float} --  use simulation time to figure out how
                                            many townhalls to build.
            townhall_build_rate {float} --  used to figure out how often to
                                            build a new townhall
            unitid              {dict}  --  "townhall_bldg"
        """
        # this defines the current maximum number of townhalls (which includes the current one) and
        # limits it to the MAX number of expandable locations minus 1 because the enemy's prob there
        max_num = min(len(self.expansion_locations_list)-1,self.max_townhalls)
        max_townhalls_rn = min(int(self.sim_time_min / self.townhall_build_rate) + 1, max_num)

        # let's do some checks to see if we should build another townhall
        need_new_bldg = self.structures(self.unitid["townhall_bldg"]).amount < max_townhalls_rn
        can_afford_bldg = self.can_afford(self.unitid["townhall_bldg"])
        bldg_being_built = self.already_pending(self.unitid["townhall_bldg"])

        # if we do need to build a new townhall, do it
        if need_new_bldg and can_afford_bldg and not bldg_being_built:
            self.logger.debug('Building a Townhall Building (%d out of %d)', \
                max_townhalls_rn, max_num)
            await self.expand_now()

    async def build_combat_structures(self):
        """Builds combat structures and all related addons. We shall build
        close to a supply building since there are more supply buildings than
        there are townhalls.

        In general, we'll only build 1 bulding per function call.

        Also, we will build "dependency buildings" first, then the scout's
        combat building, and finally, our target combat unit's building. Then,
        The target's combat building will keep being built

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            N/A

        Attributes Referenced:
            unitid  {dict}  --  "supply_bldg"
                                "combat_bldg_addons"
                                "combat_bldg"
        """
        supply_bldg = self.structures(self.unitid["supply_bldg"]).ready
        if not supply_bldg:
            return # there are no supply buildings, do not build anything!

        # else, get a random supply building
        supply_bldg = supply_bldg.random
        # make a list of buildingds to build. These buildings will be built in
        # that order....
        bldgs_to_build =    self.unitid["combat_bldg_addons"] + \
                            [self.unitid["scout_bldg"]] + \
                            [self.unitid["combat_bldg"]]

        for idx, val in enumerate(bldgs_to_build):
            # first, check if the building exists either in the pending list OR in the actual list
            # of structures. If one of the pending structures has already been built, then it is
            # ignored
            pend_bldgs = list(self.pending_combat_bldgs.keys())
            #if val in pend_bldgs:
            #    print('hmm')
            for bldg in pend_bldgs:
                if self.pending_combat_bldgs[bldg] <= self.sim_time_min:
                    # doesn't matter if it's actually there or not, remove it since time has passed
                    self.pending_combat_bldgs.pop(bldg)

            pend_bldgs = list(self.pending_combat_bldgs.keys())
            bldg_exists = self.structures(val).exists or (val in pend_bldgs)

            prev_bldg_exists = not idx or \
                                    (self.structures(bldgs_to_build[idx-1]).ready.exists and idx)
            can_afford_bldg = self.can_afford(val) # checks if we can afford to build it
            is_last_bldg = val == bldgs_to_build[-1] # checks if it's the last bldg in the list
            time_to_build_dep_bldg = not bldg_exists and can_afford_bldg

            max_bldgs_rn = int(self.sim_time_min / self.combat_bldg_build_rate)
            need_new_bldg = self.structures(val).amount < max_bldgs_rn
            time_to_keep_building_last_bldg = is_last_bldg and can_afford_bldg and need_new_bldg \
                                                and val not in pend_bldgs

            if prev_bldg_exists and (time_to_build_dep_bldg or time_to_keep_building_last_bldg):
                # make sure the previous one was made
                if not bldg_exists and not is_last_bldg:
                    self.logger.debug('Building a %s Building', val.name)
                elif is_last_bldg:
                    self.logger.debug("Building %s building #%d", \
                                                val.name, self.structures(val).amount+1)
                await self.build(val, near=supply_bldg)
                self.pending_combat_bldgs[val] = self.sim_time_min + self.wait_pending_bldg_min
                break

    async def train_combat_units(self):
        """Once the primary combat structure is built, we will start building
        at least 1 combat unit at every available combat structure (assuming
        we can afford it)

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            N/A

        Attributes Referenced:
            unitid  {dict}  --  "combat_bldg"
                                "combat"
        """
        curr_num_units = -1
        for bldg in self.structures(self.unitid["combat_bldg"]).ready.idle:
            if self.can_afford(self.unitid["combat"]) and self.supply_left > 0:
                if (self.units(self.unitid['combat']).amount % 5) == 0 and \
                                        self.units(self.unitid['combat']).amount != curr_num_units:
                    curr_num_units = self.units(self.unitid['combat']).amount
                    self.logger.debug('Training a Combat Unit, current total: %d', \
                                        self.units(self.unitid['combat']).amount)
                bldg.train(self.unitid["combat"])

        # have idle combat units go to the most vulnerable townhall
        most_vuln_townhall = self.structures(self.unitid['townhall_bldg'])
        if most_vuln_townhall:
            most_vuln_townhall = most_vuln_townhall.furthest_to(self.start_location)
        else:
            most_vuln_townhall = self.start_location
        for unit in self.units(self.unitid['combat']).idle:
            unit.move(most_vuln_townhall)

    async def engage_enemy(self):
        """In here, the bot makes the decision on how to engage the enemy.

        The bot controls idle combat units that haven't been tasked yet and can
        make them do one of four choices/actions:
        1. DISENGAGE FROM ENEMY: Run back to home base and wait for a short
           period of time (1-3 min)
        2. ENGAGE ENEMY UNITS: Target enemy units that are closest to one of
           our townhalls (randomly picked).
        3. ENGAGE ENEMY STRUCTURES: Target an enemy structure randomly.
        4. GO TO ENEMY'S START LOCATION: Send combat units to the enemy's start
           location and hope that there are enemy units/structures there to
           attack.

        The bot makes a decision based on its self.mode (that's set in config).
        Here's a brief overview on how the bot makes a choice in each mode: 
        ===========
        Rule-Based:
        ===========
        - CHOICE #1: If we have <=4 combat units
        - CHOICE #2: If we have >=6 combat units and there are visible enemy
          units.
        - CHOICE #3: If we have >=6 combat units and there are no more visible
          enemy units but there are visible enemy structures
        - CHOICE #4: If we have >=6 combat units and no enemy units or
          structures are visible (either because we've killed/destroyed
          everything or because we haven't detected anything)
        =======
        Random:
        =======
        - We just randomly make a choice b/w 1-4 using random.randrange()
        ====
        DNN:
        ====
        - We use a DNN (usually a CNN) to figure out what it thinks is the best
          choice using on our intel map's most recent data.

        Argument Keywords:
            N/A

        Raises:
            N/A

        Returns:
            N/A

        Attributes Affected:
            collect_data    {dict}  --  "training_data"

        Attributes Referenced:
            unitid          {dict}  --  "combat"
                                        "townhall_bldg"
            sim_time_min    {float}
            collect_data    {dict}  --  "exists"
                                        "training_data"
                                        "current_intel"
            model           {dict}  --  "model"
        """
        rand_wait_time_min = random.uniform(1, 3) # if we need to delay, will only delay for 1-3min
        combat_unit_rule = [6, 4] # if <4, then run back home, if >5, then engage
        choice_dict = {
            0: "No attack!",
            1: "Attack uncloaked enemy units (non-gatherer) that are close to our nexus!",
            2: "Attack enemy's townhall (or closest structure)!",
            3: "Attack enemy start location!"}
        target = {
            "found": False,
            "loc": None,
            "choice": np.zeros(4)}

        #----------------------------------------------------------------------
        # Make a Decision based on the bot's mode
        #----------------------------------------------------------------------
        if self.bot_mode == utils.BOT_MODE.RULE_BASED:
            if self.units(self.unitid["combat"]).amount <= min(combat_unit_rule):
                target["choice"][0] = 1
            elif self.units(self.unitid["combat"]).amount >= max(combat_unit_rule):
                if self.enemy_units.filter(lambda x: not x.is_cloaked).amount:
                #if self.enemy_units.amount:
                    # attack an enemy's unit
                    target["choice"][1] = 1
                elif self.enemy_structures.amount:
                    # attack an enemy's structure
                    target["choice"][2] = 1
                else:
                    # attack the enemy's start location
                    target["choice"][3] = 1
        elif self.bot_mode == utils.BOT_MODE.RANDOM:
            target["choice"][random.randrange(0, 4)] = 1
        elif self.bot_mode == utils.BOT_MODE.DNN:
            prediction = self.model['model'].predict(\
                            self.collect_data['current_intel'].reshape([-1, 176, 200, 3]))
            target["choice"][np.argmax(prediction[0])] = 1
        else:
            self.logger.error("Bot's given Mode is not handled in engage_enemy(), will not engage")
            # TODO: Is there some way to exit the bot immediately? Maybe just raise an exception?

        #----------------------------------------------------------------------
        # Based on the choice, do something
        #----------------------------------------------------------------------
        if target["choice"][0]:
            # disengage, wait for 1-3 minutes
            if self.sim_time_min > self.stay_idle_until_min:
                self.stay_idle_until_min = self.sim_time_min + rand_wait_time_min
            target["found"] = self.sim_time_min > self.stay_idle_until_min
            if self.structures(self.unitid["townhall_bldg"]):
                target["loc"] = \
                    self.structures(self.unitid["townhall_bldg"]).furthest_to(self.start_location)
            else:
                target['loc'] = self.start_location
        elif target["choice"][1]:
            # target enemy units that's closest to our furthest townhall (rel to our starting loc)
            if self.structures(self.unitid["townhall_bldg"]):
                target_townhall = \
                    self.structures(self.unitid["townhall_bldg"]).furthest_to(self.start_location)
            else:
                target_townhall = self.start_location
            e_units = self.enemy_units.filter(lambda x: not x.is_cloaked)
            #e_units = self.enemy_units
            target["found"] = len(e_units)
            target["loc"] = e_units.closest_to(target_townhall) if target['found'] else 0
        elif target["choice"][2]:
            # target enemy structures that you can see
            tgt_townhall = \
                    self.structures(self.unitid["townhall_bldg"]).furthest_to(self.start_location)
            target["found"] = len(self.enemy_structures)
            target["loc"] = self.enemy_structures.closest_to(tgt_townhall) if target['found'] else 0
            # if there's a townhall, target that first
            townhall_names = ["nexus", "commandcenter", "hatchery"]
            enemy_townhalls = self.enemy_structures.filter( \
                                lambda x: x.name.lower() in ["nexus", "commandcenter", "hatchery"])
            if enemy_townhalls and target["loc"] not in enemy_townhalls:
                # if we're not targeting a townhall, then target it
                target["loc"] = enemy_townhalls.closest_to(tgt_townhall) if target['found'] else 0
        elif target["choice"][3]:
            # target the enemy's start location
            target["found"] = True
            # fake a townhall existance
            target["loc"] = self.enemy_start_locations[0]

        #----------------------------------------------------------------------
        # If we have an action to do, then do it. Else, exit w/o doing anything
        #----------------------------------------------------------------------
        if target["found"]:
            if isinstance(target['loc'], sc2.unit.Unit):
                pos = target['loc'].position
            else:
                pos = target['loc']
            # log the result to the user
            if (target['choice'] != self.prev_target['choice']).any() or \
                                                        target['loc'] != self.prev_target['loc']:
                
                self.logger.fatal(  "Decided action: %s | Target's Location: %s", \
                                    choice_dict[np.argmax(target["choice"])], pos)
            if self.collect_data["exists"]:
                self.collect_data['training_data'].append(\
                    [target["choice"], self.collect_data['current_intel']])

            # Tell your combat units what to do
            for unit in self.units(self.unitid["combat"]):
                unit.attack(pos, queue=False)

        # finally, remember your previous attack strat to limit the number of printouts
        self.prev_target = target
