#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Author: Jiebang
# Filename: run_plot.py
# Creat Time: 2021-11-22 11:08:42 星期一
# Version: 1.0

# Description:

import math
import numpy as np

import seaborn as sns


sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)
