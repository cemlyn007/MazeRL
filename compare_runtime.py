from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import tensorflow as tf
import os

commit_new = '8ee8c71c1fd995c91559f11a50f419129b078ba7'
commit_old = '9f9ec6a3a4a88eecbbcc77834e94e1c4124644da'

root = 'runs/discrete_agent_runs'

commit_durations = []
for commit in [commit_old, commit_new]:
    durations = []
    for filename in os.listdir(root):
        if not filename.startswith(commit):
            continue
        min_wall_time = float('inf')
        max_wall_time = float('-inf')
        path = os.path.join(root, filename)
        # Iterate through each event file in the log directory
        for event_file in os.listdir(path):
            event_path = os.path.join(path, event_file)
            for e in tf.compat.v1.train.summary_iterator(event_path):
                min_wall_time = min(min_wall_time, e.wall_time)
                max_wall_time = max(max_wall_time, e.wall_time)
        duration = max_wall_time - min_wall_time
        durations.append(duration)
        print(min_wall_time, max_wall_time, max_wall_time-min_wall_time)
    commit_durations.append(durations)
    print(commit, sum(durations)/len(durations))

print(100 - (351.5388526916504 / 359.1652410030365) * 100)