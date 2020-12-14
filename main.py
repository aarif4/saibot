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
import time
import inspect
import argparse
import sc2
import core
import utils


def run_simulation(logger, user_data):
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
    # enemy parameters TODO: In future, make this flexible enough to handle DNN enemy

    # depending on which branch we're running, we may not have all of the
    # races available. So only use races that are available in the following
    # priority:
    start_time_sec = int(time.time())
    try:
        result = sc2.run_game(
            sc2.maps.get(user_data.cfg['sim_setup']['map_name']),
            [
                sc2.player.Bot(
                    user_data.cfg['player_bot']['race'],
                    eval(
                        'core.%s(user_data,"%s")' %
                        (user_data.cfg['player_bot']['race'].name, 'player_bot'))),
                sc2.player.Computer(
                    user_data.cfg['enemy_bot']['race'],
                    user_data.cfg['enemy_bot']['computer_difficulty'])
            ],
            realtime=user_data.cfg['sim_setup']['run_realtime']
        )
    except Exception as exp:
        logger.fatal('Ran into a big error running this simulation. Exiting!')
        raise exp

    stop_time_sec = int(time.time())
    time_elapsed_min = float(stop_time_sec - start_time_sec)/60.0
    return int(result == sc2.Result.Victory), time_elapsed_min

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
    # create a logger for this category of functions
    logger, _ = utils.build_logger('main')

    # parse input arguments (it's required btw)
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '-c', \
        dest='config_file', \
        type=str, \
        help='path to INI configuration file')
    args = parser.parse_args()

    # use input arguments to read a configuration file (it's required btw)
    user_data = utils.config_data(args.config_file)
    logger.info('Configuration file contents:\n%s', user_data)

    # run the requested number of trials (>0) and report the winrate
    win_loss = []
    time_taken_min = []
    num_trials = user_data.cfg['sim_setup']['num_iterations']
    for i in range(num_trials):
        s = 'Running Starcraft II Trial #%d (out of %d)' % (i+1, num_trials)
        if win_loss:
            s = '%s [Win Rate: %.2f%%] [Avg. Time Per Trial: %.2f min]' % \
                    (s, sum(win_loss)/len(win_loss)*100, sum(time_taken_min)/len(time_taken_min))

        logger.info(s)
        win_loss_val, time_taken_min_val  = run_simulation(logger, user_data)
        win_loss.append(win_loss_val)
        time_taken_min.append(time_taken_min_val)

    logger.info("=================================")
    logger.info("Done Running StarCraft II Trials!")
    logger.info("=================================")
    logger.info("Win Rate: %d/%d: %.2f%%", sum(win_loss), num_trials, sum(win_loss)/num_trials*100)
    logger.info('Total Time Taken for %d Trials:  %.4f hours (%.4f min)', \
        num_trials, sum(time_taken_min)/24, sum(time_taken_min))
    logger.info('Avg. Time Taken Per Trial:       %.4f minutes', sum(time_taken_min)/num_trials)
