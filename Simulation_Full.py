
#%%
## Simulation for the following coniditions

# Initial altitude 200km
# Final altitude 120km
# Translational and Rotational Dynamics
# initial attitude behaviour
# TLE error added too
# Output to saved is the following
# 1. Initial attitude - 3 Euler angles
# 2. Initial state in Inertial
# 3. Initial RSW error
# 4. Final state
# 5. FInal heading and flight path angle
# 6. Final altitude

#%% 
## import required packages for the code to run

import sys
from turtle import end_fill
sys.path.insert(0, "/home2/luis/tudat-bundle/.build/tudatpy")

import math as mh
import numpy as np
from matplotlib import pyplot as plt
import datetime, time
# import scipy as scp
# from scipy.spatial.transform import Rotation
from collections import defaultdict

## To check for classes in Tudat when documentation is not clear
import inspect
## use inspect.getmembers(whatever I am looking at)


from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.io import save2txt
from tudatpy.kernel.math import interpolators, root_finders


from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface


## Required for the wind model to work
from Wind_model_set_up import personal_wind_function

## Adding the TLE package as TUDAT implementation is not great
# from sgp4.api import Satrec
# from sgp4.api import jday

import skyfield.api as skfield

from multiprocessing import Pool

## For the MC simulations
import random as rnd

from mpl_toolkits import mplot3d

#%% 
## Interpolator for Aerodynamic coefficients

### For the density composition we will use bodies_created.get("Earth").atmosphere_model.density_comp_func

### The input is the following:
### 1. Altitude
### 2. Longitude
### 3. Latitude
### 4. Time

### The output is the following
###      d[0] - HE NUMBER DENSITY(CM-3)
###      d[1] - O NUMBER DENSITY(CM-3)
###      d[2] - N2 NUMBER DENSITY(CM-3)
###      d[3] - O2 NUMBER DENSITY(CM-3)
###      d[4] - AR NUMBER DENSITY(CM-3)                       
###      d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]
###      d[6] - H NUMBER DENSITY(CM-3)
###      d[7] - N NUMBER DENSITY(CM-3)
###      d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)
###      t[0] - EXOSPHERIC TEMPERATURE
###      t[1] - TEMPERATURE AT ALT
###### There are two vectors - first one is densities and the second one is temperatures

### For now we will use this function. In the future a different implementation but with the same output will be
### applied. Look at for more info
# https://github.com/tudat-team/tudat/blob/de4bce7fbf0822e532ace4cee136ed4a9e8e533d/include/tudat/astro/aerodynamics/nrlmsise00Atmosphere.h#L523 

#### Import database

## First one are the force coefficients
array_of_data = np.loadtxt("/home2/luis/Thesis/STARLINK_Database_1s_2aoa_2b_Force_FullD.txt")

## Second ones are the moments coefficients
array_of_data_2 = np.loadtxt("/home2/luis/Thesis/STARLINK_Database_1s_2aoa_2b_Moment_FullD.txt")

# #### Calculate the number of iterations for each component

# n_density = np.count_nonzero(array_of_data == array_of_data[0][0],axis=0)[0]
# n_speed = np.count_nonzero(array_of_data == array_of_data[0][1],axis=0)[1]
# n_alpha =  np.count_nonzero(array_of_data == array_of_data[0][2],axis=0)[2]
# n_sideslip = np.count_nonzero(array_of_data == array_of_data[0][3],axis=0)[3]

# Number_of_speed = np.sqrt((n_alpha*n_sideslip)/n_speed)
# Number_of_sideslip = Number_of_speed*n_speed/n_sideslip
# Number_of_alpha = n_speed/Number_of_sideslip
# Number_of_density = n_density/(Number_of_speed*Number_of_alpha*Number_of_sideslip)

# ### interval of the values for the components

# Points_of_density = np.linspace(array_of_data[0][0],array_of_data[-1][0], int(Number_of_density))
# Points_of_speed = np.linspace(array_of_data[0][1],array_of_data[-1][1], int(Number_of_speed))
# Points_of_alpha = np.linspace(array_of_data[0][2],array_of_data[-1][2], int(Number_of_alpha))
# Points_of_sideslip = np.linspace(array_of_data[0][3],array_of_data[-1][3], int(Number_of_sideslip))

# Points_of_density = np.array([1e16,1e20])
# Points_of_speed   = np.linspace(1,14,27)
# Points_of_alpha   = np.linspace(-180,180,361)
# Points_of_sideslip = np.linspace(-90,90,181)

# ## for 1s,2a,2b
Points_of_density = np.array([1e16,1e17,1e18,1e19,1e20])
Points_of_speed   = np.linspace(1,14,14)
Points_of_alpha   = np.linspace(-180,180,181)
Points_of_sideslip = np.linspace(-90,90,91)

#%% Generate dictionary
## The idea is to have a dictionary as it will be easier to find the values we require for the interpolation

class Tree(defaultdict):
    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)
        self.value = value

Force_Coeff_Dic = Tree()
Moment_Coeff_Dic = Tree()

i_tot = 0

## First loop is for the density

for i_1 in range(len(Points_of_density)):

    Current_density = Points_of_density[i_1]

    ## Second loop is for the speed ratio
    for i_2 in range(len(Points_of_speed)):

        Current_speed = Points_of_speed[i_2]

        ## Third Loop is for alpha
        for i_3 in range(len(Points_of_alpha)):

            Current_alpha = Points_of_alpha[i_3]

            ## Last Loop is for beta
            for i_4 in range(len(Points_of_sideslip)):

                Current_sideslip = Points_of_sideslip[i_4]

                Force_Coeff_Dic[Current_density][Current_speed]  \
                               [Current_alpha][Current_sideslip].value = array_of_data[i_tot,4:]

                Moment_Coeff_Dic[Current_density][Current_speed]  \
                                [Current_alpha][Current_sideslip].value = array_of_data_2[i_tot,4:]

                i_tot = i_tot + 1



#%% Generating all data and functions for the aerodynamic analysis

#### Important values for the moles

## All molar mass are in grams/mole

    
molar_mass_HE = 4.002602
molar_mass_O  = 15.999
molar_mass_N  = 14.0067
molar_mass_AR = 39.948
molar_mass_H  = 1.00784

molar_mass_list = np.array([molar_mass_HE,molar_mass_O,2*molar_mass_N,2*molar_mass_O,
                                molar_mass_AR,molar_mass_H,molar_mass_N,molar_mass_O])


Na_number     = 6.02214085774e23       # Avogadro - mol^-1
kb            = 1.3806485279e-23       # Boltzmann constant - J/K

Universal_gas_R = Na_number*kb

iteration_number_f = 0
iteration_number_m = 0

iteration_used_f = 0
iteration_used_m = 0


def Check_update_force_coeff(time):

    global iteration_number_f
    global iteration_used_f

    iteration_number_f += 1


    ## If time is the same as start of simulation update coefficients as it is
    ## the first run

    #print(iteration_number)

    if iteration_number_f < 5:
        
        new_coeff = Update_Force_Coeff(time)

        return new_coeff
    
    ## Update coefficients if iteration is number 2000

    elif round(iteration_number_f) % 100 == 0:

        new_coeff_2 = Update_Force_Coeff(time)

        return new_coeff_2
        
    else:

        force_coeff = bodies_created.get(object_of_interest).flight_conditions.aerodynamic_coefficient_interface.current_force_coefficients

        return force_coeff
    

def Update_Force_Coeff(time):


    ### First, we will import all required variables
    ### The variables are the following:
    ### Density - total -  [kg/m3]
    ### Density composition -  [/m3]
    ### Air temperature - [K]
    ### Molecule gas constant - [kJ/(kg K)]
    ### Airspeed velocity -  [m/s2]
    ### Angle of attack - [deg]
    ### Sideslip angle - [deg]

    ## First get long, lat, alt such that we can calculate values for the NRLMSISE function

    current_longitude = bodies_created.get( object_of_interest ).flight_conditions.longitude
    current_altitude = bodies_created.get( object_of_interest ).flight_conditions.altitude
    current_latitude = bodies_created.get( object_of_interest ).flight_conditions.geodetic_latitude


    current_airspeed = bodies_created.get( object_of_interest ).flight_conditions.airspeed
    angle_calculator = bodies_created.get( object_of_interest ).flight_conditions.aerodynamic_angle_calculator
    angle_of_attack = np.rad2deg(angle_calculator.get_angle(environment.angle_of_attack))
    sideslip_angle = np.rad2deg(angle_calculator.get_angle(environment.angle_of_sideslip))
    


    NRLMSISE_output = bodies_created.get("Earth").atmosphere_model.density_comp_func(current_altitude,current_longitude,
                        current_latitude,time[0])

    Density_NRLMSISE = NRLMSISE_output[0]
    T_incoming = NRLMSISE_output[1][1]

    Backup_density_NRLMSISE = Density_NRLMSISE

    Total_density = Density_NRLMSISE[5]*1000
    Density_NRLMSISE.pop(5)
    Density_composition = Density_NRLMSISE
    Density_comp_total = np.sum(Density_composition)

    ### Calculate the speed ratios
    ### Use the following equation
    ### S = airspeed_velocity/(sqrt(2*R_gas*T_inf))

    
    R_gas_list = np.ones(8)
    Speed_ratio_list = np.ones(8)
    Speed_ratio_bound = np.ones((8,2))
    Density_ratio_list = np.ones(8)

    alpha_bound = np.ones((1,2))
    sideslip_bound = np.ones((1,2))
    density_bound = np.ones((1,2))

    for i_1 in range(8):

        Current_R_gas = Universal_gas_R*1000/molar_mass_list[i_1]
        R_gas_list[i_1] = Current_R_gas

        Current_speed_ratio = current_airspeed/(np.sqrt(2*Current_R_gas*T_incoming))
        
        if Current_speed_ratio > 14:
            Current_speed_ratio = 14

        elif Current_speed_ratio <1:
            Current_speed_ratio = 1

        else:
            Current_speed_ratio = Current_speed_ratio
        
        Speed_ratio_list[i_1] = Current_speed_ratio

        ### Calculate the upper and lower bounds for the interpolation of the speed ratio

        Value_for_upper_bound = sum(i_x<Current_speed_ratio for i_x in Points_of_speed)
        Upper_speed = Points_of_speed[Value_for_upper_bound]
        Lower_speed = Points_of_speed[Value_for_upper_bound-1]
        Speed_ratio_bound[i_1][0] = Lower_speed
        Speed_ratio_bound[i_1][1] = Upper_speed


        ### Calculate the percentage of each molecule in total density
        ### Note that output from NRLMSISE is in /cm3 so you need to multuply by 10^6
        ### Calculate rho using the following formula
        ### rho = nrho/Na * molar_mass

        Current_density_particle = Density_composition[i_1]*10**(6)/Na_number * molar_mass_list[i_1]/1000
        Current_density_ratio = Current_density_particle/Total_density
        Density_ratio_list[i_1] = Current_density_ratio


        ### Calculate upper and lower bound on the angle of attack and sideslip

    Alpha_upper_bound = sum(i_x<angle_of_attack for i_x in Points_of_alpha)
    Upper_alpha = Points_of_alpha[Alpha_upper_bound]
    Lower_alpha = Points_of_alpha[Alpha_upper_bound-1]
    alpha_bound[0][0] = Lower_alpha
    alpha_bound[0][1] = Upper_alpha        
        
    sideslip_upper_bound = sum(i_x<sideslip_angle for i_x in Points_of_sideslip)
    Upper_sideslip = Points_of_sideslip[sideslip_upper_bound]
    Lower_sideslip = Points_of_sideslip[sideslip_upper_bound-1]
    sideslip_bound[0][0] = Lower_sideslip
    sideslip_bound[0][1] = Upper_sideslip

    ## Need to do a bit of a different approach for the density because at the lower values no integration is needed.
    ## For the free flow regime the minimum value is more than enough. 

    Density_upper_bound = sum(i_x<Density_comp_total for i_x in Points_of_density)

    if Density_comp_total <= 1e16:
        Upper_density = Points_of_density[Density_upper_bound]
        density_bound[0][0] = Upper_density
        density_bound[0][1] = Upper_density

    else: 
        Upper_density = Points_of_density[Density_upper_bound]
        Lower_density = Points_of_density[Density_upper_bound-1]
        density_bound[0][0] = Lower_density
        density_bound[0][1] = Upper_density





    ### Input data file
    ### For now we will do it like this until a better method is found

    Coefficients_per_molecule = np.ones((8,3))
    index_list_molecules = np.ones((8,8))
    Average_coefficients = np.ones((2,3))

    for i_density in range(2):
        
        for i_2 in range(8):


            Coefficients_l_l_l = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][0]][sideslip_bound[0][0]].value
            Coefficients_l_l_u = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][0]][sideslip_bound[0][1]].value
            Coefficients_l_u_l = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][1]][sideslip_bound[0][0]].value
            Coefficients_l_u_u = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][1]][sideslip_bound[0][1]].value
            Coefficients_u_l_l = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][0]][sideslip_bound[0][0]].value
            Coefficients_u_l_u = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][0]][sideslip_bound[0][1]].value
            Coefficients_u_u_l = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][1]][sideslip_bound[0][0]].value
            Coefficients_u_u_u = Force_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][1]][sideslip_bound[0][1]].value


            for i_3 in range(3):

                ### Interpolation of the lower bound of s

                Coefficients_I_l_l = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                                [Coefficients_l_l_l[i_3],Coefficients_l_l_u[i_3]])
                                                                            
                Coefficients_I_l_u = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                                [Coefficients_l_u_l[i_3],Coefficients_l_u_u[i_3]])
                                                                            

                Coefficients_I_l = np.interp(angle_of_attack,[alpha_bound[0][0],alpha_bound[0][1]],
                                                [Coefficients_I_l_l,Coefficients_I_l_u])
                                                                            
            

                ### Interpolation of the upper bound of s

                Coefficients_I_u_l = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                            [Coefficients_u_l_l[i_3],Coefficients_u_l_u[i_3]])
                                                                            
                Coefficients_I_u_u = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                            [Coefficients_u_u_l[i_3],Coefficients_u_u_u[i_3]])
                                                                            
                                                                            
                Coefficients_I_u = np.interp(angle_of_attack,[alpha_bound[0][0],alpha_bound[0][1]],
                                            [Coefficients_I_u_l,Coefficients_I_u_u])
                                                                            

                ### Final interpolation to find coefficients

                current_coefficients = np.interp(Speed_ratio_list[i_2],[Speed_ratio_bound[i_2][0],Speed_ratio_bound[i_2][1]],
                                                            [Coefficients_I_l,Coefficients_I_u])
                                                                            
            
                Coefficients_per_molecule[i_2][i_3] = current_coefficients*Density_ratio_list[i_2]


        Average_coefficients[i_density] = np.sum(Coefficients_per_molecule,axis=0)

    ### Interpolate coefficients for the two densities

    Average_coefficients_final = np.ones((1,3))[0]

    for i_den_coeff in range(3):
        Average_coefficients_final_arr = np.interp(Density_comp_total,[density_bound[0][0],density_bound[0][1]],
                                            [Average_coefficients[0][i_den_coeff],Average_coefficients[1][i_den_coeff]])

        Average_coefficients_final[i_den_coeff] = Average_coefficients_final_arr

    return Average_coefficients_final


#%% Set up of moment coefficients
## Update the moment coefficients 


def Check_update_mom_coeff(time):

    global iteration_number_m
    global iteration_used_m

    iteration_number_m += 1

    ## If time is the same as start of simulation update coefficients as it is
    ## the first run

    #print(iteration_number)

    if iteration_number_m < 5:

        new_coeff = Update_Mom_Coeff(time)

        return new_coeff
    
    ## Update coefficients if iteration is XXXX

    elif round(iteration_number_m) % 100 == 0:

        new_coeff_2 = Update_Mom_Coeff(time)

        return new_coeff_2

    else:

        moment_coeff = bodies_created.get(object_of_interest).flight_conditions.aerodynamic_coefficient_interface.current_moment_coefficients

        return moment_coeff
    


def Update_Mom_Coeff(time):


    ### First, we will import all required variables
    ### The variables are the following:
    ### Density - total -  [kg/m3]
    ### Density composition -  [/m3]
    ### Air temperature - [K]
    ### Molecule gas constant - [kJ/(kg K)]
    ### Airspeed velocity -  [m/s2]
    ### Angle of attack - [deg]
    ### Sideslip angle - [deg]

    ## First get long, lat, alt such that we can calculate values for the NRLMSISE function

    current_longitude = bodies_created.get( object_of_interest ).flight_conditions.longitude
    current_altitude = bodies_created.get( object_of_interest ).flight_conditions.altitude
    current_latitude = bodies_created.get( object_of_interest ).flight_conditions.geodetic_latitude


    current_airspeed = bodies_created.get( object_of_interest ).flight_conditions.airspeed
    angle_calculator = bodies_created.get( object_of_interest ).flight_conditions.aerodynamic_angle_calculator
    angle_of_attack = np.rad2deg(angle_calculator.get_angle(environment.angle_of_attack))
    sideslip_angle = np.rad2deg(angle_calculator.get_angle(environment.angle_of_sideslip))
    


    NRLMSISE_output = bodies_created.get("Earth").atmosphere_model.density_comp_func(current_altitude,current_longitude,
                        current_latitude,time[0])

    Density_NRLMSISE = NRLMSISE_output[0]
    T_incoming = NRLMSISE_output[1][1]

    Backup_density_NRLMSISE = Density_NRLMSISE

    Total_density = Density_NRLMSISE[5]*1000
    Density_NRLMSISE.pop(5)
    Density_composition = Density_NRLMSISE
    Density_comp_total = np.sum(Density_composition)

    ### Calculate the speed ratios
    ### Use the following equation
    ### S = airspeed_velocity/(sqrt(2*R_gas*T_inf))

    
    R_gas_list = np.ones(8)
    Speed_ratio_list = np.ones(8)
    Speed_ratio_bound = np.ones((8,2))
    Density_ratio_list = np.ones(8)

    alpha_bound = np.ones((1,2))
    sideslip_bound = np.ones((1,2))
    density_bound = np.ones((1,2))

    for i_1 in range(8):

        Current_R_gas = Universal_gas_R*1000/molar_mass_list[i_1]
        R_gas_list[i_1] = Current_R_gas

        Current_speed_ratio = current_airspeed/(np.sqrt(2*Current_R_gas*T_incoming))
        
        if Current_speed_ratio > 14:
            Current_speed_ratio = 14

        elif Current_speed_ratio <1:
            Current_speed_ratio = 1

        else:
            Current_speed_ratio = Current_speed_ratio
        
        Speed_ratio_list[i_1] = Current_speed_ratio

        ### Calculate the upper and lower bounds for the interpolation of the speed ratio

        Value_for_upper_bound = sum(i_x<Current_speed_ratio for i_x in Points_of_speed)
        Upper_speed = Points_of_speed[Value_for_upper_bound]
        Lower_speed = Points_of_speed[Value_for_upper_bound-1]
        Speed_ratio_bound[i_1][0] = Lower_speed
        Speed_ratio_bound[i_1][1] = Upper_speed


        ### Calculate the percentage of each molecule in total density
        ### Note that output from NRLMSISE is in /cm3 so you need to multuply by 10^6
        ### Calculate rho using the following formula
        ### rho = nrho/Na * molar_mass

        Current_density_particle = Density_composition[i_1]*10**(6)/Na_number * molar_mass_list[i_1]/1000
        Current_density_ratio = Current_density_particle/Total_density
        Density_ratio_list[i_1] = Current_density_ratio


        ### Calculate upper and lower bound on the angle of attack and sideslip

    Alpha_upper_bound = sum(i_x<angle_of_attack for i_x in Points_of_alpha)
    Upper_alpha = Points_of_alpha[Alpha_upper_bound]
    Lower_alpha = Points_of_alpha[Alpha_upper_bound-1]
    alpha_bound[0][0] = Lower_alpha
    alpha_bound[0][1] = Upper_alpha        
        
    sideslip_upper_bound = sum(i_x<sideslip_angle for i_x in Points_of_sideslip)
    Upper_sideslip = Points_of_sideslip[sideslip_upper_bound]
    Lower_sideslip = Points_of_sideslip[sideslip_upper_bound-1]
    sideslip_bound[0][0] = Lower_sideslip
    sideslip_bound[0][1] = Upper_sideslip

    ## Need to do a bit of a different approach for the density because at the lower values no integration is needed.
    ## For the free flow regime the minimum value is more than enough. 

    Density_upper_bound = sum(i_x<Density_comp_total for i_x in Points_of_density)

    if Density_comp_total <= 1e16:
        Upper_density = Points_of_density[Density_upper_bound]
        density_bound[0][0] = Upper_density
        density_bound[0][1] = Upper_density

    else: 
        Upper_density = Points_of_density[Density_upper_bound]
        Lower_density = Points_of_density[Density_upper_bound-1]
        density_bound[0][0] = Lower_density
        density_bound[0][1] = Upper_density





    ### Input data file
    ### For now we will do it like this until a better method is found

    Coefficients_per_molecule = np.ones((8,3))
    index_list_molecules = np.ones((8,8))
    Average_coefficients = np.ones((2,3))

    for i_density in range(2):
        
        for i_2 in range(8):


            Coefficients_l_l_l = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][0]][sideslip_bound[0][0]].value
            Coefficients_l_l_u = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][0]][sideslip_bound[0][1]].value
            Coefficients_l_u_l = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][1]][sideslip_bound[0][0]].value
            Coefficients_l_u_u = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][0]][alpha_bound[0][1]][sideslip_bound[0][1]].value
            Coefficients_u_l_l = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][0]][sideslip_bound[0][0]].value
            Coefficients_u_l_u = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][0]][sideslip_bound[0][1]].value
            Coefficients_u_u_l = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][1]][sideslip_bound[0][0]].value
            Coefficients_u_u_u = Moment_Coeff_Dic[density_bound[0][i_density]][Speed_ratio_bound[i_2][1]][alpha_bound[0][1]][sideslip_bound[0][1]].value


            for i_3 in range(3):

                ### Interpolation of the lower bound of s

                Coefficients_I_l_l = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                                [Coefficients_l_l_l[i_3],Coefficients_l_l_u[i_3]])
                                                                            
                Coefficients_I_l_u = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                                [Coefficients_l_u_l[i_3],Coefficients_l_u_u[i_3]])
                                                                            

                Coefficients_I_l = np.interp(angle_of_attack,[alpha_bound[0][0],alpha_bound[0][1]],
                                                [Coefficients_I_l_l,Coefficients_I_l_u])
                                                                            
            

                ### Interpolation of the upper bound of s

                Coefficients_I_u_l = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                            [Coefficients_u_l_l[i_3],Coefficients_u_l_u[i_3]])
                                                                            
                Coefficients_I_u_u = np.interp(sideslip_angle,[sideslip_bound[0][0],sideslip_bound[0][1]],
                                            [Coefficients_u_u_l[i_3],Coefficients_u_u_u[i_3]])
                                                                            
                                                                            
                Coefficients_I_u = np.interp(angle_of_attack,[alpha_bound[0][0],alpha_bound[0][1]],
                                            [Coefficients_I_u_l,Coefficients_I_u_u])
                                                                            

                ### Final interpolation to find coefficients

                current_coefficients = np.interp(Speed_ratio_list[i_2],[Speed_ratio_bound[i_2][0],Speed_ratio_bound[i_2][1]],
                                                            [Coefficients_I_l,Coefficients_I_u])
                                                                            
            
                Coefficients_per_molecule[i_2][i_3] = current_coefficients*Density_ratio_list[i_2]


        Average_coefficients[i_density] = np.sum(Coefficients_per_molecule,axis=0)

    ### Interpolate coefficients for the two densities

    Average_coefficients_final = np.ones((1,3))[0]

    for i_den_coeff in range(3):
        Average_coefficients_final_arr = np.interp(Density_comp_total,[density_bound[0][0],density_bound[0][1]],
                                            [Average_coefficients[0][i_den_coeff],Average_coefficients[1][i_den_coeff]])

        Average_coefficients_final[i_den_coeff] = Average_coefficients_final_arr

    return Average_coefficients_final


#%%
## Function for C1,C2 and C3

def C1_matrix(angle_deg,deg_0_or_rad_1):

    ## deg_or_rad = 0 if deg

    if deg_0_or_rad_1 == 0:
        angle = np.deg2rad(angle_deg)
    else:
        angle = angle_deg  

    C_1 = np.array([[1,0,0], 
                    [0,np.cos(angle),np.sin(angle)],
                    [0,-np.sin(angle),np.cos(angle)]])

    return C_1

def C2_matrix(angle_deg,deg_0_or_rad_1):

    ## deg_or_rad = 0 if deg

    if deg_0_or_rad_1 == 0:
        angle = np.deg2rad(angle_deg)
    else:
        angle = angle_deg  

    C2 = np.array([[np.cos(angle),0,-np.sin(angle)],
                   [0,1,0],
                   [np.sin(angle),0,np.cos(angle)]])

    return C2

def C3_matrix(angle_deg,deg_0_or_rad_1):

    ## deg_or_rad = 0 if deg

    if deg_0_or_rad_1 == 0:
        angle = np.deg2rad(angle_deg)
    else:
        angle = angle_deg  

    C3 = np.array([[np.cos(angle),np.sin(angle),0],
                   [-np.sin(angle),np.cos(angle),0],
                   [0,0,1]])

    return C3

#%%
## Create Environment


## load spice kernels

spice.load_standard_kernels()

## Create celestial bodies

### Bodies of interest
Celestial_bodies_to_create = ["Earth","Sun","Moon"]    

global_frame_origin = "Earth"

global_frame_orientation = "J2000" 

body_settings = environment_setup.get_default_body_settings(
        Celestial_bodies_to_create, global_frame_origin, global_frame_orientation)

## Definition of Earth parameters based on the WGS84 model

Earth_equatorial_radius = 6378137 ## In meters
Earth_flattening = 1/298.257223563
body_settings.get( "Earth" ).shape_settings = environment_setup.shape.oblate_spherical( Earth_equatorial_radius, Earth_flattening )

## Set NRLMSISE-00 model for Earth atmosphere

body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00() 

## Add HWM14 wind function
## Look at Wind_model_set_up.py for more info on the set up

custom_wind = environment_setup.atmosphere.custom_wind_model(
    personal_wind_function,
    environment.AerodynamicsReferenceFrames.vertical_frame)

body_settings.get("Earth").atmosphere_settings.wind_settings = custom_wind

## Body to be propagated

object_of_interest = "Starlink"               

body_settings.add_empty_settings( object_of_interest )


bodies_created = environment_setup.create_system_of_bodies(body_settings)


#### Create Properties of body

## Mass

bodies_created.get(object_of_interest).mass = 260.0

### Create Solar radiation pressure

radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    source_body = "Sun", 
    reference_area = 20.58, 
    radiation_pressure_coefficient = 0.2, 
    occulting_bodies = ["Earth"]
)

environment_setup.add_radiation_pressure_interface(
            bodies_created, object_of_interest, radiation_pressure_settings )



#%%
## Create aerodynamic coefficient interface settings, and add to vehicle

aero_coefficient_settings = environment_setup.aerodynamic_coefficients.custom_aerodynamic_force_and_moment_coefficients(
    Check_update_force_coeff,
    Check_update_mom_coeff,
    reference_length=1,
    reference_area=1,
    moment_reference_point = [0,0,0],
    are_coefficients_in_aerodynamic_frame = False,
    are_coefficients_in_negative_axis_direction = False,
    independent_variable_names=[environment.AerodynamicCoefficientsIndependentVariables.time_dependent]
)

#### NOTESS!!!
##  If you save in dependent variables the aerodynamic coefficients it will save
## the inputs used. So if you use body coeff it will save body coeff but if you use
## Aero coeff it will save aero coeff.

environment_setup.add_aerodynamic_coefficient_interface(
            bodies_created, object_of_interest, aero_coefficient_settings )


environment_setup.add_flight_conditions(
    bodies_created, object_of_interest,"Earth")


### Inertia Tensor Settings

## Inertia Tensor SL
Inertial_tensor_object = np.array([[1986.1,0,0],
                                  [0,2125.9,-240.4],
                                  [0,-240.0,304.1]])

bodies_created.get(object_of_interest).inertia_tensor = Inertial_tensor_object
 

 #%% 
 ## Acceleration and Torque settings


### Define accelerations acting on desired object

acceleration_settings_object = dict(
    Sun = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.cannonball_radiation_pressure()
           ],

    Moon =[
            propagation_setup.acceleration.point_mass_gravity()
          ],

    Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(24,24),
            propagation_setup.acceleration.aerodynamic()
          ]    
)

### Create global acceleration settings dictionary

acceleration_settings = {object_of_interest: acceleration_settings_object}



### Define torques per each exerting body

torque_settings_vehicle = dict(
    Earth=
    [
        propagation_setup.torque.aerodynamic()
    ]
)

# Create global torque settings dictionary
torque_settings = {object_of_interest: torque_settings_vehicle}

# Define bodies that are propagated
bodies_to_propagate = [object_of_interest]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies_created, acceleration_settings, bodies_to_propagate, central_bodies
)

# Create torque models
torque_models = propagation_setup.create_torque_models(
    bodies_created, torque_settings, bodies_to_propagate )

#%%
## Inputs

## Starlink number and TLE of interest

SL_number = 5066

line_1 = "1 55424U 23014AK  23044.37237465  .18318819  43709-5  37005-2 0  9994"
line_2 = "2 55424  69.9823  53.2471 0007250 179.4720 180.7030 16.31041054  3158"


#%%
## Generate part of initial state

satellite_skfd = skfield.EarthSatellite(line_1,line_2)
initial_pos_skfd = satellite_skfd.at(satellite_skfd.epoch).position.m
initial_vel_skfd = satellite_skfd.at(satellite_skfd.epoch).velocity.m_per_s
initial_state_trans = np.concatenate((initial_pos_skfd,initial_vel_skfd))

rotation_matrix_cart_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(
    initial_state_trans
    )

initial_state_rsw_pos = np.matmul(rotation_matrix_cart_to_rsw,initial_state_trans[0:3])
initial_state_rsw_vel = np.matmul(rotation_matrix_cart_to_rsw,initial_state_trans[3:])

print(initial_state_trans)

initial_lat,initial_long = skfield.wgs84.latlon_of(satellite_skfd.at(satellite_skfd.epoch))

initial_lat_deg = initial_lat.degrees
initial_long_deg = initial_long.degrees

initial_epoch_jd = satellite_skfd.epoch.whole + satellite_skfd.epoch.tt_fraction

simulation_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(initial_epoch_jd)


#%%
## Termination conditions

termination_variable = propagation_setup.dependent_variable.altitude( object_of_interest, "Earth" )
termination_settings = propagation_setup.propagator.dependent_variable_termination(
  dependent_variable_settings = termination_variable,
  limit_value = 120.0E3,
  use_as_lower_limit = True,
  terminate_exactly_on_final_condition=True,
   termination_root_finder_settings=root_finders.secant(
      maximum_iteration=5,
      maximum_iteration_handling=root_finders.MaximumIterationHandling.accept_result)
  )



dependent_variables_to_save = [
    propagation_setup.dependent_variable.altitude(object_of_interest,"Earth"),
    propagation_setup.dependent_variable.angle_of_attack(object_of_interest,"Earth"),
    propagation_setup.dependent_variable.sideslip_angle(object_of_interest,"Earth"),
    propagation_setup.dependent_variable.longitude(object_of_interest,"Earth"),
    propagation_setup.dependent_variable.geodetic_latitude(object_of_interest,"Earth"),
    propagation_setup.dependent_variable.aerodynamic_force_coefficients(object_of_interest,"Earth")
]


coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_56

integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        1.0,
        coefficient_set,
        np.finfo(float).eps,
        np.inf,
        10.0 ** (-10.0),
        10.0 ** (-10.0)
)


#%%
## Function

def integrator_function_MC(initial_pitch,initial_yaw,initial_roll,error_rsw1,error_rsw2,error_rsw3,error_rsw4,error_rsw5,error_rsw6):

    error_rsw = np.array([error_rsw1,error_rsw2,error_rsw3,error_rsw4,error_rsw5,error_rsw6])
        
    C_I_B = np.matmul(C3_matrix(-initial_yaw,0),
                    np.matmul(C2_matrix(-initial_pitch,0),C1_matrix(-initial_roll,0)))

    ## Rotational Propagator


    initial_state_rot_values = element_conversion.rotation_matrix_to_quaternion_entries(C_I_B)

    initial_state_rot = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    initial_state_rot[0] = initial_state_rot_values[0]
    initial_state_rot[1] = initial_state_rot_values[1]
    initial_state_rot[2] = initial_state_rot_values[2]
    initial_state_rot[3] = initial_state_rot_values[3]

    ## Add rotational velocity of the object 

    initial_state_rot[4] = 0.0
    initial_state_rot[5] = 0.0
    initial_state_rot[6] = 0.0

    ## Create rotational propagator settings

    rotational_propagator_settings = propagation_setup.propagator.rotational(
        torque_models,
        bodies_to_propagate,
        initial_state_rot,
        termination_settings,
        propagator = propagation_setup.propagator.RotationalPropagatorType.quaternions,
        )

    ## Generate new RSW initial state based on initial errors

    initial_state_pos = initial_state_rsw_pos + error_rsw[0:3]
    initial_state_vel = initial_state_rsw_vel + error_rsw[3:]

    initial_state_rsw = np.concatenate((initial_state_pos,initial_state_vel))

    ## Find transformation from RSW to inertial 

    rotation_matrix_rsw_to_cart = frame_conversion.rsw_to_inertial_rotation_matrix(
    initial_state_rsw
    )

    rotation_matrix_rsw_to_cart = np.linalg.inv(rotation_matrix_cart_to_rsw)

    ## Generate new initial states in inertial frame

    initial_state_trans_pos = np.matmul(rotation_matrix_rsw_to_cart,initial_state_rsw[0:3])
    initial_state_trans_vel = np.matmul(rotation_matrix_rsw_to_cart,initial_state_rsw[3:])
    
    initial_state_trans = np.concatenate((initial_state_trans_pos,initial_state_trans_vel))

    print(initial_state_trans)

    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state_trans,
        termination_settings,
        output_variables =  dependent_variables_to_save,
    )


    ## Create multi-type propagator

    propagator_settings_list = [translational_propagator_settings, rotational_propagator_settings]

    multi_propagator_settings = propagation_setup.propagator.multitype(
        propagator_settings_list,
        termination_settings,
        output_variables =  dependent_variables_to_save,
        print_interval = 86400.0 
        )

    # Define integrator settings

    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies_created, integrator_settings, multi_propagator_settings)
    
    if dynamics_simulator.integration_completed_successfully:
        print("Propagation successful:", dynamics_simulator.integration_completed_successfully)
    else:
        print("Shit went south !!! TERMINATING this horrible thing !!! ")

    current_states_array = result2array(dynamics_simulator.state_history)[-1]
    current_dependent_array_print = result2array(dynamics_simulator.dependent_variable_history)[-1]
    current_dependent_array = result2array(dynamics_simulator.dependent_variable_history)

    key_for_print = "_".join(str(e) for e in [np.round(initial_pitch,6),np.round(initial_yaw,6),np.round(initial_roll,6)])

    # save2txt(solution=dynamics_simulator.state_history,
    #      filename="Final_states_simulation_3_"+key_for_print+"_V2.txt",
    #      directory="/home2/luis/Thesis/Simulation_3_folder"  # default = "./" 
    #     )    
    # save2txt(solution=dynamics_simulator.dependent_variable_history,
    #      filename="Dependent_variables_simulation_3_"+key_for_print+"_V2.txt",
    #      directory="/home2/luis/Thesis/Simulation_3_folder"  # default = "./" 
    #     )

    np.savetxt("Final_states_simulation_3_"+key_for_print+".txt",
               current_states_array)
    
    np.savetxt("Dependent_variables_simulation_3_"+key_for_print+".txt",
               current_dependent_array_print)

    np.savetxt("Initial_state_simulation_3_"+key_for_print+".txt",
               initial_state_trans)

    
    

    current_weight_list = np.zeros((len(current_dependent_array[:,1])-1,3))

    for i_final in range(len(current_dependent_array[:,1])-1):

        current_coeff = current_dependent_array[i_final,6:]
        current_alpha = current_dependent_array[i_final,2]
        current_beta = current_dependent_array[i_final,3]
        current_aero_coeff = np.matmul(C3_matrix(current_beta,1),np.matmul(C2_matrix(-current_alpha,1),np.transpose(current_coeff)))
        time_step_delta = current_dependent_array[i_final+1,0] - current_dependent_array[i_final,0]
        current_weight_list[i_final,:] = np.transpose(current_aero_coeff)*time_step_delta/(current_dependent_array[-1,0] - current_dependent_array[0,0])

    average_coeff = np.sum(current_weight_list,axis = 0)

    np.savetxt("Average_coeff_simulation_3_"+key_for_print+"_V5_FullD.txt",
                average_coeff)

    return current_states_array,current_dependent_array,initial_state_trans,key_for_print

#%%
## For the MC

## generate covariance matrix

SD_rsw = np.array([0.46*1000,6.2*1000,0.14*1000,7.6,0.46,0.13])

correlation_RSW = np.array([[1.00,0.04,0.25,-0.02,-0.98,0.06],
                            [0.04,1.00,0.04,-1.00,-0.09,-0.11],
                            [0.25,0.04,1.00,-0.04,-0.30,0.00],
                            [-0.02,-1.00,-0.04,1.00,0.07,0.12],
                            [-0.98,-0.09,-0.30,0.07,1.00,-0.03],
                            [0.06,-0.11,0.00,0.12,-0.03,1.00]])

covariance_RSW = np.ones((6,6))

for i_corr in range(6):
    for j_corr in range(6):
        covariance_RSW[i_corr][j_corr] =  SD_rsw[i_corr]*SD_rsw[j_corr] * correlation_RSW[i_corr][j_corr]


mean_rsw = np.zeros(6)


## Input the angles as degrees
## We will use the random package 
np.random.seed(1234)


angles_array = np.zeros(3)
states_array = np.zeros(14)
dependent_var_array = np.zeros(13)
error_array = np.zeros(6)
initial_state_err = np.zeros(6)
input_array = np.zeros(9)
final_array = np.zeros(10)


## We will do 10 values for each angle so 10^3
## Loopfor the pitch
for i_1 in range(2000):

    print(i_1)

    initial_pitch = np.random.uniform(-180.0,180.0)
    initial_yaw   = np.random.uniform(-90.0,90.0)
    initial_roll  = np.random.uniform(-180.0,180.0)

    angles_array = np.vstack([angles_array,[initial_pitch,initial_yaw,initial_roll]])

    error_rsw = np.random.multivariate_normal(mean_rsw,covariance_RSW)
    
    error_array = np.vstack([error_array,error_rsw])
    
    input_array = np.vstack([input_array,np.concatenate([[initial_pitch,initial_yaw,initial_roll],error_rsw])])
    
    current_states_integration,current_dep_int,initial_state_w_err,key_array =  integrator_function_MC(initial_pitch,initial_yaw,initial_roll,...
                                                                                                        error_rsw[0],error_rsw[1],error_rsw[2],...
                                                                                                        error_rsw[3],error_rsw[4],error_rsw[5])
                                                                                                        
    final_array = np.vstack([final_array,current_states_integration])

np.savetxt("Input_array_simulation_3.txt",
           input_array)

np.savetxt("Keys_array_simulation_3.txt",
            angles_array)


np.savetxt("Final_array_simulation_3.txt",
            final_array)

### In case you want to use multiple cores comment out 1164 and add this lines.
### it is not perfect as you will see :D

# if __name__ == "__main__":
#   with Pool(20) as p:

#       current_states_integration,current_dep_int,initial_state_w_err,key_array =  p.starmap(integrator_function_MC,input_array)


