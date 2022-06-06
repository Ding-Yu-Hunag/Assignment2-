# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:59:36 2022

@author: USER
"""


def create_data_source(net, mode, n_timesteps=70):
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
    
    profiles = pd.DataFrame() # set up data frame in pandas
    if mode == "High Load":
        profiles['load1_p'] = net.load.p_mw[0]*1.2 +(0.1 * np.random.random(n_timesteps) * net.load.p_mw[0]) # column 1
        profiles['load2_p'] = net.load.p_mw[1]*1.2 + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[1]) # column 2
        profiles['load3_p'] = net.load.p_mw[2]*1.2 + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[2]) # column 3
        profiles['load1_q'] = net.load.q_mvar[0]*1.2 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[0]) # column 4
        profiles['load2_q'] = net.load.q_mvar[1]*1.2 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[1]) # column 5
        profiles['load3_q'] = net.load.q_mvar[2]*1.2 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[2]) # column 6
        profiles['sgen1_p'] = net.sgen.p_mw[0] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[0]) # column 7
        profiles['sgen2_p'] = net.sgen.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[1]) # column 8
        profiles['sgen3_p'] = net.sgen.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[2]) # column 9
    elif mode =="Low Load":
        profiles['load1_p'] = net.load.p_mw[0]*0.8 +(0.1 * np.random.random(n_timesteps) * net.load.p_mw[0]) # column 1
        profiles['load2_p'] = net.load.p_mw[1]*0.8 + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[1]) # column 2
        profiles['load3_p'] = net.load.p_mw[2]*0.8 + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[2]) # column 3
        profiles['load1_q'] = net.load.q_mvar[0]*0.8 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[0]) # column 4
        profiles['load2_q'] = net.load.q_mvar[1]*0.8 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[1]) # column 5
        profiles['load3_q'] = net.load.q_mvar[2]*0.8 + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[2]) # column 6
        profiles['sgen1_p'] = net.sgen.p_mw[0] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[0]) # column 7
        profiles['sgen2_p'] = net.sgen.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[1]) # column 8
        profiles['sgen3_p'] = net.sgen.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[2]) # column 9   
    elif mode =="Normal":
        profiles['load1_p'] = net.load.p_mw[0] +(0.1 * np.random.random(n_timesteps) * net.load.p_mw[0]) # column 1
        profiles['load2_p'] = net.load.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[1]) # column 2
        profiles['load3_p'] = net.load.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[2]) # column 3
        profiles['load1_q'] = net.load.q_mvar[0] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[0]) # column 4
        profiles['load2_q'] = net.load.q_mvar[1] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[1]) # column 5
        profiles['load3_q'] = net.load.q_mvar[2] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[2]) # column 6
        profiles['sgen1_p'] = net.sgen.p_mw[0] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[0]) # column 7
        profiles['sgen2_p'] = net.sgen.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[1]) # column 8
        profiles['sgen3_p'] = net.sgen.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[2]) # column 9 
    elif mode =="Generator Disconnected":
        profiles['load1_p'] = net.load.p_mw[0] +(0.1 * np.random.random(n_timesteps) * net.load.p_mw[0]) # column 1
        profiles['load2_p'] = net.load.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[1]) # column 2
        profiles['load3_p'] = net.load.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[2]) # column 3
        profiles['load1_q'] = net.load.q_mvar[0] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[0]) # column 4
        profiles['load2_q'] = net.load.q_mvar[1] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[1]) # column 5
        profiles['load3_q'] = net.load.q_mvar[2] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[2]) # column 6
        profiles['sgen1_p'] = net.sgen.p_mw[0] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[0]) # column 7
        profiles['sgen2_p'] = net.sgen.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[1]) # column 8
    elif mode =="Line Disconnected":
        profiles['load1_p'] = net.load.p_mw[0] +(0.1 * np.random.random(n_timesteps) * net.load.p_mw[0]) # column 1
        profiles['load2_p'] = net.load.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[1]) # column 2
        profiles['load3_p'] = net.load.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.load.p_mw[2]) # column 3
        profiles['load1_q'] = net.load.q_mvar[0] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[0]) # column 4
        profiles['load2_q'] = net.load.q_mvar[1] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[1]) # column 5
        profiles['load3_q'] = net.load.q_mvar[2] + (0.1 * np.random.random(n_timesteps) * net.load.q_mvar[2]) # column 6
        profiles['sgen1_p'] = net.sgen.p_mw[0] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[0]) # column 7
        profiles['sgen2_p'] = net.sgen.p_mw[1] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[1]) # column 8  
        profiles['sgen3_p'] = net.sgen.p_mw[2] + (0.1 * np.random.random(n_timesteps) * net.sgen.p_mw[2]) # column 9
    
    ds = DFData(profiles) # datasource set for panddapower
    return profiles, ds