import configparser
import os
import enum
from datetime import datetime
import sc2
import argparse
import math
import sys
import json
import logging
from termcolor import colored
import copy

class ENEMY_MODE(enum.Enum):
    COMPUTER    = 0 # DEFAULT
    #DNN         = 1 # Unused


class BOT_MODE(enum.Enum): # list(BOT_MODE.__members__)[0]
    RULE_BASED  = 0 # DEFAULT
    RANDOM      = 1
    DNN         = 2


def build_logger(log_name, fatal_name='FATAL'):
    # R,G,B,Y,M,C,W
    # "DEBUG" C
    # "INFO " G
    # "WARN " Y
    # "ERROR" R
    # "FATAL" M
    logging.addLevelName(logging.DEBUG,colored('DEBUG','cyan'))
    logging.addLevelName(logging.INFO,colored('INFO ','green'))
    logging.addLevelName(logging.WARNING,colored('WARN ','yellow'))
    logging.addLevelName(logging.ERROR,colored('ERROR','red'))
    logging.addLevelName(logging.FATAL,colored(fatal_name,'magenta'))
    #
    log_lvl = logging.DEBUG
    log = logging.getLogger(log_name)
    log.setLevel(level=log_lvl)
    #
    fmt = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s','%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(level=log_lvl)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    #
    return log, ch


class config_data():
    def __init__(self, filename=''):
        self.logger, _ = build_logger('config_data')

        self.todays_date_and_time_str = None
        self.get_todays_date_and_time()

        self.cfg = {}

        # parse the INI file
        cfg = None
        if filename is not None:
            # make sure that the file exists and is readable
            if os.path.isfile(filename) and os.access(filename, os.R_OK):
                cfg = configparser.ConfigParser()
                cfg.read(filename)
            else:
                self.logger.warning(
                    'Configuration file received is inaccessible, using default settings')
        else:
            self.logger.warning(
                'Configuration file has not been passed through, using default settings')

        # call the section parsers to use the data
        self.parse_sim_setup_section(cfg)
        self.parse_player_bot_section(cfg)
        self.parse_enemy_bot_section(cfg)
        self.parse_model_setup_section(cfg)

    def parse_sim_setup_section(self, cfg=None):
        """
        test
        """
        self.cfg['sim_setup'] = {
            'map_name':'Abyssal Reef LE',
            'num_iterations': 1,
            'run_realtime': False
        }

        # start parsing
        if cfg and 'sim_setup' in cfg:
            self.check_map_field(
                cfg,
                'sim_setup',
                'map_name',
                False)
            self.check_integer_field(
                cfg,
                'sim_setup',
                'num_iterations',
                False,
                1,
                1e6)
            self.check_bool_field(
                cfg,
                'sim_setup',
                'run_realtime',
                False)

    def parse_player_bot_section(self, cfg=None):
        """
        test
        """
        self.cfg['player_bot'] = {
            'mode':BOT_MODE.RULE_BASED,  # REQUIRED
            'race': sc2.Race.Protoss,       # REQUIRED
            'model_location': '',
            'save_training_data': False,
            'training_data_dir': '',
            'plot_map_intel': False,
            'max_num_workers': 65,
            'use_worker_scout': False
        }

        # start parsing
        if cfg and 'player_bot' in cfg:
            self.check_enum_field(
                cfg,
                'player_bot',
                'mode',
                'upper',
                'BOT_MODE',
                True)
            self.check_enum_field(
                cfg,
                'player_bot',
                'race',
                'capitalize',
                'sc2.Race',
                True)
            self.check_dir_field(
                cfg,
                'player_bot',
                'model_location',
                self.cfg['player_bot']['mode'] in [BOT_MODE.DNN],
                '',
                True) # already exists
            self.check_bool_field(
                cfg,
                'player_bot',
                'save_training_data',
                False)
            self.check_dir_field(
                cfg,
                'player_bot',
                'training_data_dir',
                self.cfg['player_bot']['save_training_data'],
                '{}/{}'.format('training_data', self.todays_date_and_time_str),
                False) # should not already exist
            self.check_bool_field(
                cfg,
                'player_bot',
                'plot_map_intel',
                False)
            self.check_integer_field(
                cfg,
                'player_bot',
                'max_num_workers',
                False,
                15,
                200)
            self.check_bool_field(
                cfg,
                'player_bot',
                'use_worker_scout',
                False)

    def parse_enemy_bot_section(self, cfg=None):
        """
        test
        """
        self.cfg['enemy_bot'] = {
            'mode':ENEMY_MODE.COMPUTER,  # REQUIRED
            'race': sc2.Race.Protoss,    # REQUIRED
            'computer_difficulty': sc2.Difficulty.Easy # Required iff mode="COMPUTER"
        }

        # start parsing
        if cfg and 'enemy_bot' in cfg:
            self.check_enum_field(
                cfg,
                'enemy_bot',
                'mode',
                'upper',
                'ENEMY_MODE',
                True)
            self.check_enum_field(
                cfg,
                'enemy_bot',
                'race',
                'capitalize',
                'sc2.Race',
                True)
            self.check_enum_field(
                cfg,
                'enemy_bot',
                'computer_difficulty',
                'capitalize',
                'sc2.Difficulty',
                self.cfg['enemy_bot']['mode'] == ENEMY_MODE.COMPUTER)

    def parse_model_setup_section(self, cfg=None):
        """
        test
        """
        self.cfg['model_setup'] = {
            'gen_model': False, # REQUIRED
            'train_data_dir': '', # REQUIRED
            'save_dir': '', # REQUIRED
            'model_details': '', # Required iff gen_model=True
            'verbose': False,
            'max_num_datasets': 100,
            'test_data_ratio': 0.3,
            'batch_size': 50,
            'learning_rate': 1e-3,
            'num_epochs': 10,
            'increment': 20
        }

        if cfg and 'model_setup' in cfg:
            self.check_bool_field(
                cfg,
                'model_setup',
                'gen_model',
                True)
            self.check_dir_field(
                cfg,
                'model_setup',
                'train_data_dir',
                True,
                '',
                True) # already exists
            self.check_dir_field(
                cfg,
                'model_setup',
                'save_dir',
                self.cfg['model_setup']['gen_model'],
                '{}/{}'.format('models', self.todays_date_and_time_str),
                False) # should not already exist
            self.check_json_field(
                cfg,
                'model_setup',
                'model_details',
                self.cfg['model_setup']['gen_model'])
            self.check_bool_field(
                cfg,
                'model_setup',
                'verbose',
                False)
            self.check_integer_field(
                cfg,
                'model_setup',
                'max_num_datasets',
                False,
                1,
                math.inf)
            self.check_float_field(
                cfg,
                'model_setup',
                'test_data_ratio',
                False,
                0,
                1)
            self.check_integer_field(
                cfg,
                'model_setup',
                'batch_size',
                False,
                1,
                math.inf)
            self.check_float_field(
                cfg,
                'model_setup',
                'learning_rate',
                False,
                1e-9,
                math.inf)
            self.check_integer_field(
                cfg,
                'model_setup',
                'num_epochs',
                False,
                1,
                math.inf)
            self.check_integer_field(
                cfg,
                'model_setup',
                'increment',
                False,
                1,
                math.inf)

    def make_build_path(self, dirname=''):
        """
        """
        return

    def get_todays_date_and_time(self):
        """
        finds today's date and time and sets self.todays_data_and_time_str
        to the following:

        "YYYY-MM-DD_HH-MM-SS"

        Does not return anything
        """
        # get today's date and time
        today = datetime.today()
        # format today's date and time to be used in build folders
        self.todays_date_and_time_str = \
            "%04d-%02d-%02d_%02d-%02d-%02d" % \
                (   today.year, \
                    today.month, \
                    today.day, \
                    today.hour, \
                    today.minute, \
                    today.second )

    def check_enum_field(self, cfg, section_name, field_name, change_fmt, enum_class_str, required):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        change_fmt:     can only be "lower", "upper", or "capitalize". Never empty
        enum_class_str: string notation of a class
        required:       bool to force that field_name to exist
        """
        try:
            cfg_val = cfg[section_name][field_name].strip("'").strip('"')

            good_vals = eval('list(%s.__members__)' % (enum_class_str,))
            good_lowercase_vals = \
                [a.lower() for a in good_vals]
            val_idx = [idx for idx,val in enumerate(good_lowercase_vals) if cfg_val.lower() == val]
            if val_idx:
                idx = val_idx[0]
                self.cfg[section_name][field_name] = \
                    eval('%s["%s"]' % (enum_class_str, good_vals[idx]))
            else:
                raise KeyError
        except KeyError as exp: # field does not exist in cfg
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value "%s"',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name].name)
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be set to %s. Cannot run!',
                    section_name,
                    field_name,
                    eval('list(%s.__members__)' % (enum_class_str,)))
                raise exp
        except AttributeError as exp: # bad enum value given
            self.logger.error(
                '[%s]\'s "%s" field\'s value can only be one of the following: %s',
                section_name,
                field_name,
                eval('list(%s.__members__' % (enum_class_str,)))
            raise exp

    def check_bool_field(self, cfg, section_name, field_name, required):
        """
        cfg:                is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        """
        try:
            self.cfg[section_name][field_name] = \
                cfg[section_name].getboolean(field_name)
        except KeyError: # field does not exist in cfg
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value %s',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be set to %s. Cannot run!',
                    section_name,
                    field_name,
                    ["True", "False"])
                raise exp

    def check_integer_field(self, cfg, section_name, field_name, required, low, high):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        low:            lowest integer value to have (inclusive)
        high:           highest integer value to have (not including)
        [low, high)
        """
        try:
            self.cfg[section_name][field_name] = cfg[section_name].getint(field_name)
            if not (high > self.cfg[section_name][field_name] >= low):
                raise ValueError
        except KeyError as exp:
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value %d',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be set to [%d,%d). Cannot run!',
                    section_name,
                    field_name,
                    low,
                    high)
                raise exp
        except ValueError as exp: # invalid range
            self.logger.error(
                '[%s]\'s "%s" field must be set to [%d,%d). Cannot run!',
                section_name,
                field_name,
                low,
                high)
            raise exp

    def check_float_field(self, cfg, section_name, field_name, required, low, high):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        low:            lowest integer value to have (including)
        high:           highest integer value to have (excluding)
        [low, high)
        """
        try:
            self.cfg[section_name][field_name] = cfg[section_name].getfloat(field_name)
            if not (high > self.cfg[section_name][field_name] >= low):
                raise ValueError
        except KeyError as exp:
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value %d',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be set to [%d,%d). Cannot run!',
                    section_name,
                    field_name,
                    low,
                    high)
                raise exp
        except ValueError as exp: # invalid range
            self.logger.error(
                '[%s]\'s "%s" field must be set to [%d,%d). Cannot run!',
                section_name,
                field_name,
                low,
                high)
            raise exp

    def check_dir_field(self, cfg, section_name, field_name, required, subfolder, already_exists):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        subfolder:      additional path to add to this string
        """
        try:
            self.cfg[section_name][field_name] = cfg[section_name][field_name].strip("'").strip('"')
            if self.cfg[section_name][field_name]:
                if self.cfg[section_name][field_name][-1] in ['\\', '/']:
                    self.cfg[section_name][field_name] = self.cfg[section_name][field_name][:-1]
            else: # is empty string
                raise ValueError

            # make folder from path
            if subfolder:
                self.cfg[section_name][field_name] = \
                    "{}/{}".format(self.cfg[section_name][field_name], subfolder)

            if required and not os.access(self.cfg[section_name][field_name], os.F_OK):
                os.makedirs(self.cfg[section_name][field_name])
        except KeyError as exp:
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value "%s"',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be set to a valid path. Cannot run!',
                    section_name,
                    field_name)
                raise exp
        except ValueError as exp:
            if required:
                self.logger.error(
                    '[%s]\'s "%s" field cannot be empty string. Cannot run!',
                    section_name,
                    field_name)
                raise exp
        except FileExistsError as exp:
            if already_exists:
                pass # should be fine
            elif not required:
                self.logger.warning(
                    '[%s]\'s "%s" field\'s ("%s") directory already exists.',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.warning(
                    '[%s]\'s "%s" field\'s ("%s") dir already exists. This should not happen. Cannot run!',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
                raise exp

    def check_json_field(self, cfg, section_name, field_name, required):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        """
        try:
            self.cfg[section_name][field_name] = cfg[section_name][field_name].strip("'").strip('"')

            # parse json
            self.cfg[section_name][field_name] = json.loads(self.cfg[section_name][field_name])
        except KeyError as exp:
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value "%s"',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be a valid JSON string. Cannot run!',
                    section_name,
                    field_name)
                raise exp
        except ValueError as exp:
            self.logger.error(
                '[%s]\'s "%s" field\'s JSON string could not be parsed successfully. Cannot run!',
                section_name,
                field_name)
            raise exp

    def check_map_field(self, cfg, section_name, field_name, required):
        """
        cfg:            is configparser object
        section_name:   string
        field_name:     string
        required:       bool to force that field_name exist
        """
        try:
            self.cfg[section_name][field_name] = cfg[section_name][field_name].strip("'").strip('"')
        except KeyError as exp:
            if not required:
                self.logger.warning(
                    '[%s]\'s "%s" field doesn\'t exist. Using default value "%s"',
                    section_name,
                    field_name,
                    self.cfg[section_name][field_name])
            else:
                self.logger.error(
                    '[%s]\'s "%s" field doesn\'t exist. Must be a valid StarCraft II map name. Cannot run!',
                    section_name,
                    field_name)
                raise exp
        try:
            val = sc2.maps.get(self.cfg[section_name][field_name])
        except KeyError as exp:
            self.logger.error(
                '[%s]\'s "%s" field value ("%s") is NOT a valid StarCraft II map name. Cannot run!',
                section_name,
                field_name,
                self.cfg[section_name][field_name])
            raise exp

    def __str__(self):
        enum_locs = [
            ['player_bot','mode'],
            ['player_bot','race'],
            ['enemy_bot','mode'],
            ['enemy_bot','race'],
            ['enemy_bot','computer_difficulty']]
        cfg_str = copy.deepcopy(self.cfg)
        for val in enum_locs:
            i = val[0]
            j = val[1]
            cfg_str[i][j] = str(cfg_str[i][j])
        return json.dumps(cfg_str, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '-c', \
        dest='config_file', \
        type=str, \
        help='path to INI configuration file')

    args = parser.parse_args()
    data = config_data(args.config_file)
    print('Configuration file contents:\n%s' % (data,))
