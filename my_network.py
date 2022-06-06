# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:58:33 2022

@author: USER
"""

def my_network(mode):
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
    
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu = "flat", init_va_degree = "dc", calculate_voltage_angles=True)

    # bus
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    b4 = pp.create_bus(net, 110)
    b5 = pp.create_bus(net, 110)
    b6 = pp.create_bus(net, 110)
    b7 = pp.create_bus(net, 110)
    b8 = pp.create_bus(net, 110)
    b9 = pp.create_bus(net, 110)

    # slack bus
    pp.create_ext_grid(net, b1) 
    
    # lines
    pp.create_line(net, b1, b4, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b4, b5, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b4, b9, 10, "149-AL1/24-ST1A 110.0")
    
    if mode == "High Load" or mode == "Low Load" or mode == "Generator Disconnected" or mode == "Normal":
        pp.create_line(net, b5, b6, 10, "149-AL1/24-ST1A 110.0")
        
    pp.create_line(net, b8, b9, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b6, b7, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b2, b8, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b7, b8, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b6, 10, "149-AL1/24-ST1A 110.0")
    print(net.line)
    
    # load
    pp.create_load(net, b5, p_mw=90., q_mvar=30., name='load1')
    pp.create_load(net, b7, p_mw=100., q_mvar=35., name='load2')
    pp.create_load(net, b9, p_mw=125., q_mvar=50., name='load3')
    
    # generator
    if mode == "Generator Disconnected": 
        pp.create_sgen(net, b1, p_mw=0., q_mvar=0., name='sgen1')
        pp.create_sgen(net, b2, p_mw=163., q_mvar=0., name='sgen2')
        print(net.sgen)
    else:
        pp.create_sgen(net, b1, p_mw=0., q_mvar=0., name='sgen1')
        pp.create_sgen(net, b2, p_mw=163., q_mvar=0., name='sgen2')
        pp.create_sgen(net, b3, p_mw=85., q_mvar=0., name='sgen3')
        print(net.sgen)
    # pp.plotting.simple_plot(net)
    return net