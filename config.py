#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
import logging
import time
import envs.config_env as config_env
import agents.config_agents as config_agent
import os
import sys 
flags = tf.compat.v1.flags

config_env.config_env(flags)
config_agent.config_agent(flags)

flags.DEFINE_integer("seed", 1, "Seed value")
flags.DEFINE_string("folder", "", "Folder for the log file")

# Parse flags
flags.FLAGS(sys.argv)

# Default folder handling
log_folder = flags.FLAGS.folder if flags.FLAGS.folder else "default_folder"

# Ensure necessary directories exist
log_dir = os.path.join("./results/logs/", log_folder)
pred_dir = os.path.join("./results/preds/", log_folder)
nn_dir = os.path.join("./results/nn/", log_folder)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(nn_dir, exist_ok=True)

# Create result file with a given filename
now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
file_name = config_env.get_filename() + "-" + config_agent.get_filename()
file_name += "-seed-" + str(flags.FLAGS.seed) + "-" + s_time

logger = logging.getLogger('Logger')
logger.propagate = False
logger.setLevel(logging.INFO)

log_filename = os.path.join(log_dir, f"r-{file_name}.txt")
logger_fh = logging.FileHandler(log_filename)

logger_fm = logging.Formatter('%(asctime)s\t%(message)s')
logger_fh.setFormatter(logger_fm)
logger.addHandler(logger_fh)

nn_filename = os.path.join(nn_dir, f"nn-{file_name}")

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty': 0,
    'wall': 1,
    'predator': 2,
    'prey': 3
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
