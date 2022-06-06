# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:41:34 2022

@author: USER
"""

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

import matplotlib.pyplot as plt 

from timeseries import timeseries
from my_network import my_network
from create_data_source import create_data_source
from create_controllers import create_controllers
from create_output_writer import create_output_writer
from plot import plot
from eq_distance import eq_distance


#########################################################################################################
##### Generate data for 5 different modes
mode = "High Load"
output_dir_high_load = os.path.join(tempfile.gettempdir(), "time_series_example", mode)
print("Results can be found in your local temp folder: {}".format(output_dir_high_load))
if not os.path.exists(output_dir_high_load):
    os.mkdir(output_dir_high_load)
timeseries(output_dir_high_load, mode)

mode = "Low Load"
output_dir_low_load = os.path.join(tempfile.gettempdir(), "time_series_example", mode)
print("Results can be found in your local temp folder: {}".format(output_dir_low_load))
if not os.path.exists(output_dir_low_load):
    os.mkdir(output_dir_low_load)
timeseries(output_dir_low_load, mode)

mode = "Normal"
output_dir_normal = os.path.join(tempfile.gettempdir(), "time_series_example", mode)
print("Results can be found in your local temp folder: {}".format(output_dir_normal))
if not os.path.exists(output_dir_normal):
    os.mkdir(output_dir_normal)
timeseries(output_dir_normal, mode)

mode = "Generator Disconnected"
output_dir_generator_disconnected = os.path.join(tempfile.gettempdir(), "time_series_example", mode)
print("Results can be found in your local temp folder: {}".format(output_dir_generator_disconnected))
if not os.path.exists(output_dir_generator_disconnected):
    os.mkdir(output_dir_generator_disconnected)
timeseries(output_dir_generator_disconnected, mode)

mode = "Line Disconnected"
output_dir_line_disconnected = os.path.join(tempfile.gettempdir(), "time_series_example", mode)
print("Results can be found in your local temp folder: {}".format(output_dir_line_disconnected))
if not os.path.exists(output_dir_line_disconnected):
    os.mkdir(output_dir_line_disconnected)
timeseries(output_dir_line_disconnected, mode)

# Put data together
dir_list = [output_dir_high_load, output_dir_low_load, output_dir_normal, output_dir_generator_disconnected, output_dir_line_disconnected]



#########################################################################################################
##### Reading data from time series result 
# High Load
vpu_high_load_file = os.path.join(output_dir_high_load, "res_bus", "vm_pu.xls")
read_vpu_high_load = pd.read_excel(vpu_high_load_file, index_col=0)
angle_high_load_file = os.path.join(output_dir_high_load, "res_bus", "va_degree.xls")
read_angle_high_load = pd.read_excel(angle_high_load_file, index_col=0)
high_load = pd.concat([read_vpu_high_load, read_angle_high_load], axis=1, keys = ['vpu', 'angle'])
high_load['mode'] = 'High Load'


# Low Load
vpu_low_load_file = os.path.join(output_dir_low_load, "res_bus", "vm_pu.xls")
read_vpu_low_load = pd.read_excel(vpu_low_load_file, index_col=0)
angle_low_load_file = os.path.join(output_dir_low_load, "res_bus", "va_degree.xls")
read_angle_low_load = pd.read_excel(angle_low_load_file, index_col=0)
low_load = pd.concat([read_vpu_low_load, read_angle_low_load], axis=1, keys = ['vpu', 'angle'])
low_load['mode'] = 'Low Load'

# Normal
vpu_normal_file = os.path.join(output_dir_normal, "res_bus", "vm_pu.xls")
read_vpu_normal = pd.read_excel(vpu_normal_file, index_col=0)
angle_normal_file = os.path.join(output_dir_normal, "res_bus", "va_degree.xls")
read_angle_normal = pd.read_excel(angle_normal_file, index_col=0)
normal = pd.concat([read_vpu_normal, read_angle_normal], axis=1, keys = ['vpu', 'angle'])
normal['mode'] = 'Normal'

# Generator Disconnected
vpu_generator_disconnected_file = os.path.join(output_dir_generator_disconnected, "res_bus", "vm_pu.xls")
read_vpu_generator_disconnected = pd.read_excel(vpu_generator_disconnected_file, index_col=0)
angle_generator_disconnected_file = os.path.join(output_dir_generator_disconnected, "res_bus", "va_degree.xls")
read_angle_generator_disconnected = pd.read_excel(angle_generator_disconnected_file, index_col=0)
generator_disconnected = pd.concat([read_vpu_generator_disconnected, read_angle_generator_disconnected], axis=1, keys = ['vpu', 'angle'])
generator_disconnected['mode'] = 'Generator Disconnected'

# Line Disconnected
vpu_line_disconnected_file = os.path.join(output_dir_line_disconnected, "res_bus", "vm_pu.xls")
read_vpu_line_disconnected = pd.read_excel(vpu_line_disconnected_file, index_col=0)
angle_line_disconnected_file = os.path.join(output_dir_line_disconnected, "res_bus", "va_degree.xls")
read_angle_line_disconnected = pd.read_excel(angle_line_disconnected_file, index_col=0)
line_disconnected = pd.concat([read_vpu_line_disconnected, read_angle_line_disconnected], axis=1, keys = ['vpu', 'angle'])
line_disconnected['mode'] = 'Line Disconnected'

All_data = pd.concat([high_load, low_load, normal, generator_disconnected, line_disconnected])

inputs = All_data.to_numpy()



#########################################################################################################
##### Generate data for Kmeans (delete type tag in colunm 19)
inputs_termporary = []
for i in inputs:
    listt = i.tolist()
    del listt[-1]
    inputs_termporary.append(listt)
    # i[-1].remove
inputs_plot = np.array(inputs_termporary)
inputs_kmeans = np.array(inputs_termporary)



#########################################################################################################
##### plot
plot(inputs_plot[0:70], "High Load")
plot(inputs_plot[70:140], "Low Load")
plot(inputs_plot[140:210], "Normal Operation")
plot(inputs_plot[210:280], "Generator Disconnected")
plot(inputs_plot[280:350], "Line Disconnected")



#########################################################################################################
##### Normalization
# In order to normalize the data, we find the max and min values for each of
# the  columns in the inputs
max = np.amax(inputs_kmeans, axis=0)
min = np.amin(inputs_kmeans, axis=0)

# Using numpy array magic, we go through the inputs array and regularize all values
inputs_kmeans = (inputs_kmeans-min)/(max-min)

for i in range(len(inputs_kmeans)): # Adjust slack bus data to avoid devided by zero
        inputs_kmeans[i,0] = 1  # Slack bus voltage = 1
        inputs_kmeans[i,9] = 0  # Slack bus angle = 0
print(inputs_kmeans)



#########################################################################################################
##### K means
num_of_Kmeans = 10 # run 10 times of Kmeans to average 
num_times_cost_list = [] # list that store the cost of different k for ten times
for num in range(num_of_Kmeans):
    cost_list = []
    for k in [1,2,3,4,5,6,7]: # try k = 1~7 to see the cost
        mean_list = np.zeros((k, inputs_kmeans.shape[1]))   # list that store centroid of each cluster (will be updated every loop)
        new_mean_list = np.zeros((k, inputs_kmeans.shape[1]))# list that store centroid of each cluster (will be updated every loop)
        mean_threshold = 1e-4
        cost = 0   
        
        for i in range(k): # depends on k, create k initial point
            init_mean = inputs_kmeans[np.random.randint(inputs_kmeans.shape[0]),:]*0.99 # find initial centroid from the data randomly
            mean_list[i] = init_mean
        save = mean_list.tolist() # meet some strange errors, therefore I create a list to save centroid before it changes
        
        count = 0
        condition = True
        while condition:
            count += 1 # number of iteration to complete Kmeans
            cluster_list = []
            mean_diff_list = []   
            for i in range(k): 
                cluster_list.append([]) # create k clusters
                mean_diff_list.append([]) # create a list to store distance between old&new centroid
                
            for data in inputs_kmeans:
                distance_list = []
                for mean in mean_list:          
                    distance_list.append(eq_distance(data, mean)) # Calculate distance between data point and centroid
                
                nan_index_array = np.isnan(distance_list) # Occurs some nan value, find nan and put it as zero
                nan_index_list = nan_index_array.tolist()
                for nan in range(len(nan_index_list)):
                    if nan_index_list[nan] == True:
                        distance_list[nan] = 0
                    elif nan == False:
                        pass
                index = distance_list.index(np.min(distance_list))
                cluster_list[index].append(data.tolist()) 
             
            for i in range(k): # Calculate new mean for eacg cluster
                new_mean_list[i] = np.average(np.array(cluster_list[i]), axis=0)  # Calculate new centroid 
                mean_diff_list[i] = eq_distance(new_mean_list[i].tolist(), save[i]) # distance between new & old centroid
                
            if np.sum(mean_diff_list) < mean_threshold: # If centoid is not moving
                # print("Finish clustering")
                condition = False
            elif count > 2000: # If already iterates more than 2000 times
                condition = False
            else: # centroid moves to new one
                # print("else")
                mean_list = new_mean_list
                save = mean_list.tolist()            
        
        # Check
        for i in range(k):  
            print(len(cluster_list[i])) # see how many datas in each cluster  
        print("k=", k) # when k is..?
        print("count = ", count)   # see how many times it converge
         
        for i in range(k): # Calculate the cost
            for data in cluster_list[i]:
                cost += eq_distance(data, mean_list[i])
        cost_list.append(cost)
        print("cost=", cost)
        
    num_times_cost_list.append(cost_list)

# Calculate the average of cost for ten times
num_times_cost_array = np.array(num_times_cost_list)
num_times_cost = np.average(num_times_cost_array, axis=0).tolist()


# plot 10 elbow function
for plot_time in range(num_of_Kmeans):
    plt.plot([1,2,3,4,5,6,7], num_times_cost_list[plot_time])
plt.xlabel('k')
plt.ylabel('Cost of each time')
plt.title("Cost for diffenet times")
plt.show()

# plot average elbow function
print(num_times_cost)
plt.plot([1,2,3,4,5,6,7], num_times_cost)
plt.xlabel('k')
plt.ylabel('Average Cost')
plt.title("Average Cost")
plt.show()



#########################################################################################################
##### KNN preparation

# training set & testing set
All_data_training = pd.concat([high_load[0:50], low_load[0:50], normal[0:50], generator_disconnected[0:50], line_disconnected[0:50]])
All_data_testing = pd.concat([high_load[50:70], low_load[50:70], normal[50:70], generator_disconnected[50:70], line_disconnected[50:70]])
inputs_training = All_data_training.to_numpy()
inputs_testing = All_data_testing.to_numpy()


# Normalization
max[0] = 0.01    # to avoid devided by 0
max[9] = 0.01    # to avoid devided by 0
for inputss in inputs_training:
    new_input = (inputss[0:18]-min)/(max-min)
    inputss[0:18] = new_input
    inputss[0] = 1  # put the number back (slack bus doesn't really contribute to clustering, becuz distance of that dimension is always = 0)
    inputss[9] = 0  # put the number back (slack bus doesn't really contribute to clustering, becuz distance of that dimension is always = 0)
for inputss in inputs_testing:
    new_input = (inputss[0:18]-min)/(max-min)
    inputss[0:18] = new_input
    inputss[0] = 1  # put the number back (slack bus doesn't really contribute to clustering, becuz distance of that dimension is always = 0)
    inputss[9] = 0  # put the number back (slack bus doesn't really contribute to clustering, becuz distance of that dimension is always = 0)


# The correct answer of training data
KNN_answer_training = []
# Answer of KNN fo training / testing data
for data in inputs_training:
    KNN_answer_training.append(data[18])
# print(KNN_answer_training)

# The correct answer of training data
KNN_answer_testing = []
for data in inputs_testing:
    KNN_answer_testing.append(data[18])
# print(KNN_answer_testing)
    


#########################################################################################################
# KNN Training
error_list = []
accuracy_list = []

for k in range(1,10): # Test k = 1~10
    data_cluster_list = [] # List to store result o clustering: which mode 
    for data in inputs_training:
        distance_list = [] # list store distance to every point
        for compare in inputs_training:          
            distance_list.append(eq_distance(data[0:18], compare[0:18])) # Calculate distance
            sort_distance = sorted(range(len(distance_list)), key = lambda kk : distance_list[kk]) # Sorting
            
    
        k_nearest_point = [] # List save k nearest point
        k_nearest_point_distance = [] # List save distance of j nearest point
        for i in range(k): 
            k_nearest_point_distance.append(distance_list[sort_distance[i]])
            k_nearest_point.append(inputs_training[sort_distance[i]])
            

        # 5 modes
        count_HL = 0
        count_LL = 0
        count_N = 0
        count_GD = 0
        count_LD = 0
        count_list = []
        state_list = ["High Load", "Low Load", "Normal", "Generator Disconnected", "Line Disconnected"] 
        for knn in k_nearest_point:
            if knn[18] == "High Load":
                count_HL += 1
            if knn[18] == "Low Load":
                count_LL += 1
            if knn[18] == "Normal":
                count_N += 1
            if knn[18] == "Generator Disconnected":
                count_GD += 1
            if knn[18] == "Line Disconnected":
                count_LD += 1
        count_list = [count_HL, count_LL, count_N, count_GD, count_LD] # Clustering result: How many counts for each mode
        
        index = count_list.index(np.max(count_list)) # Which mode with most counts is the clustering result
        data_cluster_list.append(state_list[index])
    
    error = 0
    for i in range(len(data_cluster_list)):
        if data_cluster_list[i] != KNN_answer_training[i]:
            error += 1 # Calculate how many errors by comparing with correct answer
    accuracy = (k + 1 - error)/(k+1) # Calculate accuracy for k = 1~10
    error_list.append(error)
    accuracy_list.append(accuracy) 

print(accuracy_list) 
# print(error_list)



#########################################################################################################
### KNN Testing
k=5   # set k = 5
data_cluster_list = [] # List to store result o clustering: which mode 
for data in inputs_testing:
    distance_list = [] # list store distance to every point
    for compare in inputs_testing:          
        distance_list.append(eq_distance(data[0:18], compare[0:18])) # Calculate distance
        sort_distance = sorted(range(len(distance_list)), key = lambda kk : distance_list[kk]) # Sorting
        

    k_nearest_point = [] # List save k nearest point
    k_nearest_point_distance = [] # List save distance of j nearest point
    for i in range(k): 
        k_nearest_point_distance.append(distance_list[sort_distance[i]])
        k_nearest_point.append(inputs_testing[sort_distance[i]])
    
    
    # 5 modes
    count_HL = 0
    count_LL = 0
    count_N = 0
    count_GD = 0
    count_LD = 0
    count_list = []
    state_list = ["High Load", "Low Load", "Normal", "Generator Disconnected", "Line Disconnected"] 
    for knn in k_nearest_point:
        if knn[18] == "High Load":
            count_HL += 1
        if knn[18] == "Low Load":
            count_LL += 1
        if knn[18] == "Normal":
            count_N += 1
        if knn[18] == "Generator Disconnected":
            count_GD += 1
        if knn[18] == "Line Disconnected":
            count_LD += 1
    count_list = [count_HL, count_LL, count_N, count_GD, count_LD] # Clustering result: How many counts for each mode
    
    index = count_list.index(np.max(count_list)) # Which mode with most counts is the clustering result
    data_cluster_list.append(state_list[index])

error = 0
for i in range(len(data_cluster_list)):
    if data_cluster_list[i] != KNN_answer_testing[i]:
        error += 1  # Calculate how many errors by comparing with correct answer     
accuracy = (k + 1 - error)/(k+1) # Calculate accuracy


# print("error = ", error)
print("accuracy = ", accuracy)

















