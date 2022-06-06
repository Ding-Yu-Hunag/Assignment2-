# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:54:17 2022

@author: USER
"""

def timeseries(output_dir, mode):
    import os
    import numpy as np
    import pandas as pd
    import tempfile
    import math

    import pandapower as pp
    from pandapower.timeseries import DFData
    from pandapower.timeseries import OutputWriter
    from pandapower.timeseries.run_time_series import run_timeseries
    from pandapower.control import ConstControl
    from pandapower.plotting import simple_plot
    
    from my_network import my_network
    from create_data_source import create_data_source
    from create_controllers import create_controllers
    from create_output_writer import create_output_writer
    
    # 1. create test net
    net = my_network(mode)

    # 2. create (random) data source
    n_timesteps = 70
    profiles, ds = create_data_source(net, mode, n_timesteps) # two outputs of the function
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, mode, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, mode, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps)