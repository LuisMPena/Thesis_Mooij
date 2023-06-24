from pickle import FALSE
import sys
from turtle import end_fill
#sys.path.insert(0, "/home/luist/tudat-bundle/cmake-build-release-wsl-notocar/tudatpy/")
sys.path.insert(0, "/home/luist/tudat-bundle/cmake-build-release-wsl/tudatpy/")


##### Import the proj. It is required for the HWM14 model. For some reason sometimes it doesnt read it properly.
##### For more infor look itup online I do not understnad why it happens
import os

os.environ['PROJ_LIB'] = "/home/luist/anaconda3/envs/tudat-bundle/bin/proj"

import numpy as np
from matplotlib import pyplot as plt
import datetime, time
import requests
import pandas as pd
import pyhwm2014 as hwm14model
import warnings


##### HWM14 model downloaded from the following link 
##### https://github.com/rilma/pyHWM14 
##### Verified against results from 
##### https://kauai.ccmc.gsfc.nasa.gov/instantrun/hwm 



start_time = time.time()

###### Set up the code to get the Ap Index file from celestrack and make it a lookup table

url_celes_ap = "https://celestrak.com/SpaceData/sw19571001.txt"

# url_celes_ap = "/home/luist/sw19571001.txt"

#### Whole Data file just in case

celes_ap = pd.read_table(url_celes_ap, sep="\t+", engine='python',skiprows=16)

#### Measured values dictionary generation
#### Should stay updated if the code from above is ran

celes_ap_measured = np.array([np.ones(33)])
key_list = np.array([])

for index_i, row_i in celes_ap.iterrows():

        ##### Make sure that the file is "cut" at the end of the observed values
        if row_i["BEGIN OBSERVED"] == "END OBSERVED":

                index_end_observed = index_i
                break

        else:

                ### Select current row and split it into each different values from the file
                current_values =  row_i["BEGIN OBSERVED"]
                array_of_data_str = np.array(current_values.split())

                ### Create the key for the dictionary
                ### For the key the following format is used in string!!
                ### Format is year-month-day ---- If month and day are single a zero is infront
                ### Eg. 2002-02-03 would be a valid key
                current_key = array_of_data_str[0] + "-" + array_of_data_str[1] + "-" + array_of_data_str[2]
                key_list = np.append(key_list, str(current_key))

                ### Difference between array_of_data_str and array_of_data 
                ### the second one has values as int and float and not only str

                array_of_data = np.ones(len(array_of_data_str))

                ### This loop is because for some reason if using only float the year sometime ends with .0 instead of nothing
                ### To fix this and make the string generation easier then just make the initial 23 values int and then the rest float
                ### Might give errors in the future so keep it into account

                for i_1 in range(len(array_of_data_str)):
                        if i_1 <23:
                                array_of_data[i_1] = int(array_of_data_str[i_1])
                        else:
                                array_of_data[i_1] = float(array_of_data_str[i_1])

                if len(array_of_data) == 33:
                        celes_ap_measured=  np.append(celes_ap_measured, [array_of_data], axis=0)
                else:
                        array_of_data = np.insert(array_of_data,27,0)
                        celes_ap_measured=  np.append(celes_ap_measured, [array_of_data], axis=0)



##### Generation of dictionary with all years, months, days and values

celes_ap_measured = celes_ap_measured[1:,:]
celes_ap_measured_dict = dict(zip(key_list,celes_ap_measured[:,3:]))

#### Predicted values generation
#### Same explanation as before except for a single line

celes_ap_predicted_val = pd.read_table(url_celes_ap, sep="\t+", engine='python',skiprows=16+index_end_observed+4)

celes_ap_predicted = np.array([np.ones(33)])
key_list_predicted = np.array([])

for index_i2, row_i2 in celes_ap_predicted_val.iterrows():
        if row_i2["BEGIN DAILY_PREDICTED"] == "END DAILY_PREDICTED":
                index_end_predicted = index_i2
                break
        else:
                current_values =  row_i2["BEGIN DAILY_PREDICTED"]
                array_of_data_str = np.array(current_values.split())
                current_key = array_of_data_str[0] + "-" + array_of_data_str[1] + "-" + array_of_data_str[2]
                key_list_predicted = np.append(key_list_predicted, str(current_key))
                array_of_data = np.ones(len(array_of_data_str))

                for i_2 in range(len(array_of_data_str)):
                        if i_2 <23:
                                array_of_data[i_2] = int(array_of_data_str[i_2])
                        else:
                                array_of_data[i_2] = float(array_of_data_str[i_2])
                
                if len(array_of_data) == 33:
                        celes_ap_measured=  np.append(celes_ap_measured, [array_of_data], axis=0)
                else:
                        array_of_data = np.insert(array_of_data,27,0)
                        celes_ap_measured=  np.append(celes_ap_measured, [array_of_data], axis=0)
                
                ### This line adds a value of 0 because for some reason the predicted vaslues are missing a value. 
                ### This means that a dictionary cannot be created with both measured and predicted unless you fix it. 

                # array_of_data = np.insert(array_of_data, 27,0)

                # celes_ap_predicted=  np.append(celes_ap_predicted, [array_of_data], axis=0)



##### Generation of dictionary with all years, months, days and values

celes_ap_predicted = celes_ap_predicted[1:,:]

celes_ap_predicted_dict = dict(zip(key_list_predicted,celes_ap_predicted[:,3:]))

### Dictionary with all the values

celes_ap_total = np.append(celes_ap_measured,celes_ap_predicted,axis=0)
key_list_total = np.append(key_list,key_list_predicted)
celes_ap_total_dic = dict(zip(key_list_total,celes_ap_total[:,3:]))



#%% Wind function generation

def personal_wind_function(alt,long,lat,current_time):

        ### This function calculated the velocity in u and v axis from the spacecraft using the HWM14 model
        ### Inputs are:
        ### Altitude - in m because it gets divided afterwards
        ### Geodetic longitude - in degrees
        ### Geodetic latitude - in degrees
        ### Current time - in seconds from 2000-01-01 00:00:00

        #### Calculate the new date time as current_time is in seconds

        initial_time = datetime.datetime(2000,1,1)
        new_date = initial_time + datetime.timedelta(seconds = current_time)
        new_date_str = str(initial_time + datetime.timedelta(seconds = current_time))
        current_year = int(new_date_str[0:4])
        current_month = int(new_date_str[5:7])
        current_day = int(new_date_str[8:10])
        current_hour = int(new_date_str[11:13])
        current_time = int(new_date_str[14:16])
        current_second = int(new_date_str[17:19])
        percentage_hour = (current_time*60 + current_second)/(3600)
        ut_time = current_hour + percentage_hour

        #### Look for Ap Index value from Table 

        new_date_key = new_date_str[0:10]

        data_string_current_date = celes_ap_total_dic[new_date_key]
        #### Depending on the hour of the day you get a different Ap 3h index value

        if current_hour <= 3:
                ap_index_wind = data_string_current_date[11]
        elif 3 < current_hour <=6:
                ap_index_wind = data_string_current_date[12]
        elif 6 < current_hour <=9:
                ap_index_wind = data_string_current_date[13]
        elif 9 < current_hour <=12:
                ap_index_wind = data_string_current_date[14]
        elif 12 < current_hour <=15:
                ap_index_wind = data_string_current_date[15]
        elif 15 < current_hour <=18:
                ap_index_wind = data_string_current_date[16]
        elif 18 < current_hour <=21:
                ap_index_wind = data_string_current_date[17]
        else:
                ap_index_wind = data_string_current_date[18]


        #### Need ot fix the latitude and longitude.
        #### TUDAT gives values in radians but degrees are required
        #### Also, Long goes from 0 to 360 in this software while TUDAT is -180 to 180

        if long < 0:
                long_deg = np.rad2deg(2*np.pi + long)
        else:
                long_deg = np.rad2deg(long)
        
        lat_deg = np.rad2deg(lat)

        #### CHECK THAT YOU ARE NOT GOING INTO THE FUTURE!!!

        todays_date = datetime.datetime.now()
        
        difference_w_current_time = (new_date - todays_date).total_seconds()
                
        if difference_w_current_time < 0:
                pass
        else:
                warnings.warn("Using predicted values for solar activity as date goes beyond today",
                              ImportWarning)

        #### Run HWM14 model

        hwm14_model_output = hwm14model.HWM14( alt = alt/1000, altlim = [alt/1000,alt/1000], altstp=1,
                                         ap=[-1, ap_index_wind], glat=lat_deg,glon=long_deg, 
                                         day=current_day, option=1, ut=ut_time, verbose=False, 
                                         year=current_year )



        return [hwm14_model_output.Uwind[0],hwm14_model_output.Vwind[0],0]


print("--- %s seconds ---" % (time.time() - start_time))
