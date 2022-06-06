# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:07:05 2022

@author: USER
"""

def create_output_writer(net, mode, time_steps, output_dir):
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
    
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop  
    if mode == "High Load":
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')    
    elif mode =="Low Load":
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')  
    elif mode =="Normal":
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')
    elif mode =="Generator Disconnected":
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')  
    elif mode =="Line Disconnected":
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')  
    return ow