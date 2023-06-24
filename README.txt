-----------------------------------
--- How to install the software ---
-----------------------------------


- First. Install TUDAT following the instructions from the TUDAT GitHub
	- Make sure you use version 2.11.1.dev3 as it will be easier to edit the C++ code to add the dependencies required
- Second. Once TUDAT is installed, two modifications need to be made. Look at TUDAT_Modification_file
	- Now generate the conda environment for tudat-bundle
- Third. Install the HWM14 model. Instructions are on HWM14_Installation_file
- Fourth. Install skyfield - https://rhodesmill.org/skyfield/
- Fifth. Download the database of coefficients, both force and moment coefficients. 
- Sixth. Enjoy :D

- In case you want to run the code from the faculty server, you need to use a connection between the server and the laptop. 
- I used visual studio and followed the steps in Install_SSH_VS


------------------------
--- File explanation ---
------------------------

- Wind_model_set_up.py
	- Explains how to obtain the wind model for the TUDAT to run. 
	- Note - it takes an online file to run - if wanted, you can download the text file and add it to the directory to reduce time
	- Should take about 30s to run, loaded once per simulation only

- Simulation_Full.py
	- Contains the full simulation
	- It "SHOULD" run without any issues
	- Need to change a couple of lines to the correct directory, such as the import sys and the force/moment files
	- Returns final state, position, velocity and quaternions, with some other dependent variables, of the simulation