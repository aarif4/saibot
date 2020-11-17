"""
    This is the main script that runs our StarCraft bot against a computer
    player. Depending on the availability of StarCraft bots in core/, this
    script shall either start running or yield an exception indicating that
    you're using a "bad branch".

    Please make sure you use requirements.txt to configure your Python venv
    and ensure that you have installed StarCraft 2
    (https://www.blizzard.com/en-us/download) in the typical location
    in your filesystem.

    If you have installed StarCraft 2 in a "non-typical" path, then please
    edit burny-sc2/sc2/paths.py to point to the new location of StarCraft 2.

    Here are the expected paths for a couple of popular OSes:
        "Windows": "C:/Program Files (x86)/StarCraft II"
        "Darwin":  "/Applications/StarCraft II"
        "Linux":   "~/StarCraftII"

    When StarCraft 2 runs the simulation, another window will open up. This is
    a diagnostic tool used to indicate your bot's army and whatever portion of
    the enemy's army that has been visible. This tool helps us get a broader
    perspective of the battlefield along with the ratio of army units and
    resources.
"""
import sys
import inspect
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
import core

MAX_ITER = 1 # maximum number of times to run simulation

def run_simulation():
    """
    This function simply runs the StarCraft API and simulate a game with
    our bot versus the computer. Depending on whether we're in a dev branch
    or in mainline, certain StarCraft race bots may or may not exist

    Therefore, we check if that class exists and if so, then we'll use it

    Inputs:
        N/A

    Output:
        N/A
    """
    # enemy parameters
    enemy_race       = Race.Protoss
    realtime_flag    = False
    enemy_difficulty = Difficulty.Easy

    # depending on which branch we're running, we may not have all of the
    # races available. So only use races that are available in the following
    # priority:
    if class_exists('Protoss'):
        run_game(
            maps.get("Abyssal Reef LE"), 
            [
                Bot(core.Protoss().get_race(), core.Protoss()),
                Computer(enemy_race, enemy_difficulty)
            ], 
            realtime=realtime_flag )
    elif class_exists('Terran'):
        run_game(
            maps.get("Abyssal Reef LE"), 
            [
                Bot(core.Terran().get_race(), core.Terran()),
                Computer(enemy_race, enemy_difficulty)
            ], 
            realtime=realtime_flag )
    elif class_exists('Zerg'):
        run_game(
            maps.get("Abyssal Reef LE"), 
            [
                Bot(core.Zerg().get_race(), core.Zerg()),
                Computer(enemy_race, enemy_difficulty)
            ], 
            realtime=realtime_flag )
    else:
        errmsg = "No StarCraft Bots available. "
        errmsg = errmsg + "Please contact @aarif4 for details"
        raise Exception(errmsg)


def class_exists(class_name):
    """
    This function basically checks 'core' module for what Starcraft Bots are
    present and whether the user's requested class_name exists in 'core'

    Inputs:
        class_name: nonempty string with a class name

    Ouptuts:
        Returns a boolean flag indicating existance of class
            True  = Yes, the class you're looking for exists, use it
            False = No, the class you're looking for is unavailable
    """
    # get all of the modules in core
    import_modules = inspect.getmembers(sys.modules[core.__name__])
    # extract classes that aren't built-in
    available_classes = [i[0] for i in import_modules if i[0][0] != '_']

    return class_name in available_classes


if __name__ == "__main__":
    for i in range(MAX_ITER):
        print('=== Running Starcraft II Trial #%d (out of %d)' % (i+1, MAX_ITER))
        run_simulation()
