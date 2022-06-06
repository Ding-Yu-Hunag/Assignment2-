# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:05:24 2022

@author: USER
"""


def create_controllers(net, mode, ds):
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
    
    if mode =="Generator Disconnected":      
        ConstControl(net, element='load', variable='p_mw', element_index=[0],
                     data_source=ds, profile_name=["load1_p"])
        ConstControl(net, element='load', variable='p_mw', element_index=[1],
                     data_source=ds, profile_name=["load2_p"])
        ConstControl(net, element='load', variable='p_mw', element_index=[2],
                     data_source=ds, profile_name=["load3_p"])
        
        ConstControl(net, element='load', variable='q_mvar', element_index=[0],
                     data_source=ds, profile_name=["load1_q"])
        ConstControl(net, element='load', variable='q_mvar', element_index=[1],
                     data_source=ds, profile_name=["load2_q"])
        ConstControl(net, element='load', variable='q_mvar', element_index=[2],
                     data_source=ds, profile_name=["load3_q"])
        
        ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                      data_source=ds, profile_name=["sgen1_p"])
        ConstControl(net, element='sgen', variable='p_mw', element_index=[1],
                      data_source=ds, profile_name=["sgen2_p"])
    else:
        ConstControl(net, element='load', variable='p_mw', element_index=[0],
                     data_source=ds, profile_name=["load1_p"])
        ConstControl(net, element='load', variable='p_mw', element_index=[1],
                     data_source=ds, profile_name=["load2_p"])
        ConstControl(net, element='load', variable='p_mw', element_index=[2],
                     data_source=ds, profile_name=["load3_p"])
        
        ConstControl(net, element='load', variable='q_mvar', element_index=[0],
                     data_source=ds, profile_name=["load1_q"])
        ConstControl(net, element='load', variable='q_mvar', element_index=[1],
                     data_source=ds, profile_name=["load2_q"])
        ConstControl(net, element='load', variable='q_mvar', element_index=[2],
                     data_source=ds, profile_name=["load3_q"])
        
        ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                      data_source=ds, profile_name=["sgen1_p"])
        ConstControl(net, element='sgen', variable='p_mw', element_index=[1],
                      data_source=ds, profile_name=["sgen2_p"])
        ConstControl(net, element='sgen', variable='p_mw', element_index=[2],
                      data_source=ds, profile_name=["sgen3_p"])