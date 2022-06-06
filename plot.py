# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:34:22 2022

@author: USER
"""

def plot(data, title):
    import matplotlib.pyplot as plt 
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    for k in range(70):
        plt.scatter(data[k][0], data[k][9], color = color_list[0])
        plt.scatter(data[k][1], data[k][10], color = color_list[1])
        plt.scatter(data[k][2], data[k][11], color = color_list[2])
        plt.scatter(data[k][3], data[k][12], color = color_list[3])
        plt.scatter(data[k][4], data[k][13], color = color_list[4])
        plt.scatter(data[k][5], data[k][14], color = color_list[5])
        plt.scatter(data[k][6], data[k][15], color = color_list[6])
        plt.scatter(data[k][7], data[k][16], color = color_list[7])
        plt.scatter(data[k][8], data[k][17], color = color_list[8])
    plt.xlabel('Voltage (pu)')
    plt.ylabel('Angle (deg.)')
    plt.title(title)
    plt.legend(["bus1", "bus2", "bus3", "bus4", "bus5", "bus6", "bus7", "bus8","bus9"], loc ="best")
    plt.show()