'''
Python lab for learning how to build a flow rate estimator based on echogram data and
more specifically echogram data points selected/segmented by machine learning.

author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chardet 
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from config import *
import SBL_physics as sbl

import echopype as ep
import netCDF4 as n


def get_gas_flux(TS_AVG,PSI,RHO_GAS):
    '''
    Function that estimates the gas flux from average target strength, bubble rise speed and gas density.

    Inputs:
    TS_AVG: Average target strength of the bubble plume (dB)
    PSI: "Average" vertical rise velocity of the bubbles (m/s)
    RHO_GAS: Density of the gas at depth (kg/m3)

    Outputs:
    GAS_FLUX: Estimated gas flux (m3/s)
    '''

    #Convert target strength from dB to linear scale
    TS_LINEAR = 10**(TS_AVG/10)

    #Estimate gas flux using a simplified model
    #This is a placeholder formula and should be replaced with a more accurate model
    GAS_FLUX = RHO_GAS * TS_LINEAR * PSI #m3/s

    return GAS_FLUX


if __name__ == "__main__":
    
    '''
    Key parameters:
    TS_AVG: Average target strength of the bubble plume (dB)
    PSI: "Average" vertical rise velocity of the bubbles (m/s)
    BSD_PDF: Probability density function of the bubble size distribution

    '''
    
    #some testing
    # I am testing backscattering cross-section calculations for different bubble sizes
    TS_AVG = -40.4  #dB
    PSI = 0.02733 #m/s, What is this???
    DEPTH = 220  #m
    density_input = 1025  #kg/m3
    vector_size = 1000 
    bubble_sizes = np.logspace(-5, -1, vector_size)  # Bubble sizes from 10 micrometers to 10 centimeters
    STATIC_PRESSURE = sbl.get_hydrostatic_pressure(DEPTH,rho_sw=density_input) + 101325  #Pa
    TEMPERATURE = 4  #degC
    SALINITY = 35  #KG/ m3
    RHO_GAS = []
    BSD_PDF = []
    BRS_AVG = [] #Should be same size as BSD_PDF and give rising speed for each bubble size bin in BSD_PDF
    RS_PDF = [] #Bubble rise speed probability density function
    RESONANT_RADII = [] #Resonant bubble radius at echosounder frequency (38kHz) for each depth
    DELTA_DAMPING = [] #Dimensionless damping coefficient for each bubble size bin in BSD_PDF
    BS_CROSSECTION = [] #Backscattering cross-section for each bubble size bin in BSD_PDF

    #GET VARIABLES

    # --- BUBBLE SIZE DISTRIBUTION PROBABILITY DENSITY FUNCTION --- #
    bsd_data_file_loc = DATA_DIR / 'bubble_characteristics' / 'BSD_example.txt'
    with open(bsd_data_file_loc, 'rb') as f:
        result = chardet.detect(f.read())
    BSD = pd.read_csv(bsd_data_file_loc, delimiter='\t', encoding=result['encoding'], header=None).values
    BSD_PDF = sbl.generate_bsd_pdf(BSD, min=0.001, max=0.006, vectorsize=vector_size)

    # --- BUBBLE RISE SPEED FOR EACH BUBBLE SIZE BIN IN BSD_PDF --- #
    min_bs,max_bs,vec_size = np.min(BSD_PDF[:,1]),np.max(BSD_PDF[:,1]),vector_size
    BRS_AVG = sbl.get_brsm(bubble_size_range=[min_bs,max_bs],
                           temperature=TEMPERATURE,
                           salinity=35,
                           model = 'Fan1990',
                           tuning = [1.4,1.6],
                           resolution=vec_size)
    BRS_PDF= sbl.get_rising_speed_pdf(BSD_PDF,BRS_AVG,resolution=vec_size)

    #plot bubble rise speeds vs bubble sizes

    plt.figure()
    plt.plot(BSD_PDF[:,1],BSD_PDF[:,0], label='BSD PDF', color='blue')
    plt.xlabel('Bubble Size (m)')
    plt.ylabel('Probability Density',color='blue')
    plt.tick_params(axis='y',labelcolor='blue')
    plt.twinx()
    plt.plot(BSD_PDF[:,1], BRS_AVG[:,1], color='orange', label='BRS AVG')  #convert mm to m
    plt.xlabel('Bubble Size (m)')
    plt.ylabel('Rise velocity (m/s)', color='orange')
    plt.tick_params(axis='y', labelcolor='orange')
    plt.title('Bubble Size Distribution and Rise Speed')
    plt.show()

    #################################################################################################
    # --- CALCULATE THE BUBBLE BACKSCATTERING CROSS-SECTION SPECTRUM FOR DIFFERENT BUBBLE SIZES --- #
    ##### ###########################################################################################
    # at 220 meter depth and for a 38 kHz echosounder

    DEPTH = 220  #m
    density_input = 1025  #kg/m3
    vector_size = 1000 
    bubble_sizes = np.logspace(-5, -1, vector_size)  # Bubble sizes from 10 micrometers to 10 centimeters
    freq_echosounder = 38000  #1/s
    STATIC_PRESSURE = sbl.get_hydrostatic_pressure(DEPTH,rho_sw=density_input) + 101325  #Pa
    TEMPERATURE = 4  #degC
    SALINITY = 35  #KG/ m3
    RHO_GAS = []
    #BSD_PDF = []
    BRS_AVG = [] #Should be same size as BSD_PDF and give rising speed for each bubble size bin in BSD_PDF
    RS_PDF = [] #Bubble rise speed probability density function
    RESONANT_RADII = [] #Resonant bubble radius at echosounder frequency (38kHz) for each depth
    DELTA_DAMPING = [] #Dimensionless damping coefficient for each bubble size bin in BSD_PDF
    BS_CROSSECTION = [] #Backscattering cross-section for each bubble size bin in BSD_PDF

    # --- DENSITY OF GAS AT DEPTH --- #
    pressure = sbl.get_hydrostatic_pressure(DEPTH) + 101325  #Pa
    RHO_GAS = sbl.get_gas_density(temperature=TEMPERATURE, pressure=pressure, gas_type='methane', method='vdw')  #kg/m3
    
    # --- SOUND VELOCITY AT DEPTH --- #
    sound_speed = sbl.get_sound_speed(temperature=TEMPERATURE, salinity=SALINITY, depth=DEPTH)

    # --- RESONANT BUBBLE RADIUS AT ECHOSOUNDER FREQUENCY (38kHz) AS A FUNCTION OF DEPTH --- #
    depths = np.linspace(0, 500, vector_size)  # Depths from 0 to 500 meters
    RESONANT_RADII = np.zeros(len(depths))
    for i, depth in enumerate(depths):
        pressure_depth = sbl.get_hydrostatic_pressure(depth) + 101325
        RESONANT_RADII[i] = sbl.get_resonant_bubble_size(freq_echosounder=freq_echosounder,
                                                          density_water=density_input,
                                                          specific_heat_ratio=1.31,
                                                          pressure_static=pressure_depth)

    # --- DELTA DAMPING COEFFICIENT --- #
    DELTA_DAMPING = np.zeros(vector_size)

    # calculate the Minnaert frequency for each bubble size bin
    freq_Minnaert = np.zeros(vector_size)
    for i in range(vector_size):
        freq_Minnaert[i] = sbl.get_Minnaert_frequency(radius_bubble=bubble_sizes[i],
                                                       density_water=density_input,
                                                       specific_heat_ratio=1.4,  # This is a value from some other source (not Veloso) 1.31
                                                       pressure_static=sbl.get_hydrostatic_pressure(DEPTH,rho_sw=density_input) + 101325
                                                       )  # Convert from rad/s to Hz

    for i in range(vector_size):
        DELTA_DAMPING[i] = sbl.get_delta_damping_coefficient(radius_bubble=bubble_sizes[i],
                                                              freq_Minnaert=freq_Minnaert[i],
                                                              diffusivity_thermal=sbl.get_methane_properties(property='thermal diffusivity'),
                                                              viscosity_shear=0.01107*0.001,
                                                              viscosity_bulk=1.3*0.01107*0.001,
                                                              density_water=density_input,
                                                              speed_sound = sound_speed,
                                                              specific_heat_ratio=1.31)

    # --- BACKSCATTERING CROSS-SECTION FOR EACH BUBBLE SIZE BIN IN BSD_PDF --- #
    resonant_radius_at_depth = RESONANT_RADII[np.abs(depths - DEPTH).argmin()]
    BS_CROSSECTION = sbl.get_bubble_backscattering_crosssection(bubble_sizes,
                                                                radius_resonant_bubble=resonant_radius_at_depth,
                                                                freq_echosounder=freq_echosounder,
                                                                speed_sound=sound_speed,
                                                                delta_coeff=DELTA_DAMPING)

    BS_CROSSECTION = BS_CROSSECTION #Convert to m²

    ###################
    # --- TESTING --- #
    ###################

    # --- ESTIMATE MASS FLUX FROM A GIVEN AVERAGE TARGET STRENGTH --- #

    TS_AVG = -40.4  #dB
    TS_AVG = -39.47
    vector_size = 1000
    DEPTH = 220  #m
    density_water = 1025  #kg/m3
    density_gas = sbl.get_gas_density(temperature=TEMPERATURE, pressure=STATIC_PRESSURE, gas_type='methane', method='vdw')  #kg
    density_gas = 1
    speed_sound = sbl.get_sound_speed(temperature=TEMPERATURE, salinity=SALINITY, depth=DEPTH)
    # use bsd from the data (already loaded above)
    BSD_PDF[:,0] = BSD_PDF[:,0] / np.sum(BSD_PDF[:,0])
    # test a BSD with only 6 mm bubbles
    BSD_PDF[:,0] = np.ones_like(BSD_PDF[:,0])/len(BSD_PDF[:,0])
    BSD_PDF[:,1] = np.ones_like(BSD_PDF[:,1]) * 0.003

    min_bs,max_bs,vec_size = np.min(BSD_PDF[:,1]),np.max(BSD_PDF[:,1]),vector_size
    BRS_AVG = sbl.get_brsm(bubble_size_range=[min_bs,max_bs],
                           temperature=TEMPERATURE,
                           salinity=35,
                           model = 'Fan1990',
                           tuning = [1.4,1.6],
                           resolution=vec_size)
    BRS_PDF= sbl.get_rising_speed_pdf(BSD_PDF,BRS_AVG,resolution=vec_size)

    plt.figure()
    plt.plot(BSD_PDF[:,1],BSD_PDF[:,0], label='BSD PDF', color='blue')
    plt.ylabel('Probability Density')
    plt.twinx()
    plt.plot(BSD_PDF[:,1], BRS_AVG[:,1], color='orange', label='BRS AVG')  #convert mm to m
    plt.xlabel('Bubble Size (m)')
    plt.ylabel('Rise velocity (m/s)')
    plt.title('Bubble Size Distribution and Rise Speed')
    plt.legend()
    plt.show()


    #Calculate backscattering cross section for bubbles in BSD_PDF
    bubble_sizes_bsd = BSD_PDF[:,1]
    freq_echosounder = 38000  #Hz
    speed_sound = sbl.get_sound_speed(temperature=TEMPERATURE, salinity=SALINITY, depth=DEPTH)
    resonant_radius_at_depth = RESONANT_RADII[np.abs(depths - DEPTH).argmin()]
    delta_coeff_bsd = DELTA_DAMPING[np.abs(bubble_sizes - bubble_sizes_bsd[:,None]).argmin(axis=1)]
    BS_CROSSECTION_BSD = sbl.get_bubble_backscattering_crosssection(bubble_sizes_bsd,
                                                                    radius_resonant_bubble=resonant_radius_at_depth,
                                                                    freq_echosounder=freq_echosounder,
                                                                    speed_sound=speed_sound,
                                                                    delta_coeff=delta_coeff_bsd)
    # plot the BS_CROSSECTION_BSD vs bubble sizes in BSD_PDF
    plt.figure()
    plt.loglog(bubble_sizes_bsd, BS_CROSSECTION_BSD)
    plt.xlabel('Bubble Size (m)')
    plt.ylabel('Backscattering Cross Section (m²)')
    plt.title('Backscattering Cross Section vs Bubble Size')
    plt.grid()
    plt.show()

    #get the BRS for the bubble sizes in BSD_PDF
    BRS_AVG_bsd = np.zeros((len(bubble_sizes_bsd),2))
    BRS_AVG_bsd[:,1] = bubble_sizes_bsd
    for i in range(len(bubble_sizes_bsd)):
        BRS_AVG_bsd[i,0] = BRS_AVG[np.abs(BRS_AVG[:,0]-bubble_sizes_bsd[i]).argmin(),0]

    #BRS_AVG[:,1] = 0.249
    # Calculate the flux ampplifier PSI
    PSI = sbl.get_flux_amplifier_PSI(radius_bubble = BSD_PDF[:,1],
                             bubble_size_PDF = BSD_PDF[:,0],
                             backscattering_crossection_array = BS_CROSSECTION_BSD,
                             rise_velocity_array = BRS_AVG[:,1],
                             speed_sound = speed_sound,
                             transmit_pulse_length=1024/1000000)  #0.512 \mus)
    print(f"Calculated PSI: {PSI:.6f} m/s") 

    # Estimate gas flux
    gas_flux_estimate = sbl.get_mass_flux(PSI, density_gas,TS_AVG) #density set to 1 to get volumetric flow
    gas_flux_estimate = gas_flux_estimate
    #print(f"Estimated Gas Flux: {gas_flux_estimate:.6f} m³/s")
    #or in ml/min
    gas_flux_estimate_ml_min = gas_flux_estimate * 1e6 * 60  # Convert m³/s to ml/min
    print(f"Estimated Gas Flux: {gas_flux_estimate_ml_min:.2f} ml/min")
    print(f"PSI: {PSI:.6f} m/s")

    ###########################
    # --- Make some plots --- #
    ###########################

    # --- Bubble Size Distribution PDF and Rise Speed, yyplot --- #
    '''
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Bubble Size (m)')
    ax1.set_ylabel('Probability Density Function', color=color)
    ax1.plot(BSD_PDF[:,1], BSD_PDF[:,0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, np.max(BSD_PDF)*1.1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Bubble Rise Speed (m/s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(BSD_PDF[:,1], BRS_AVG[:,1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, np.max(BRS_AVG[:,1])*1.1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Bubble Size Distribution PDF and Rise Speed')
    plt.show()
    '''

    # --- Resonant Bubble Radius vs Depth and Backscattering Cross-Section vs Bubble Size --- #

    plt.figure()
    plt.plot(depths,RESONANT_RADII*1000)  # Convert to mm for better readability
    plt.ylabel('Resonant Bubble Radius (mm)')
    plt.xlabel('Depth (m)')
    plt.title('Resonant Bubble Radius vs Depth at 38 kHz')
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.loglog(bubble_sizes, BS_CROSSECTION)  #Convert to mm for better readability
    #plot vertical dashed lines at 0.41 mm, 1 mm and 6 mm
    plt.axvline(x=0.00041, color='k', linestyle='--',)
    plt.axvline(x=0.001, color='k', linestyle='--')
    plt.axvline(x=0.006, color='k', linestyle='--')
    #plot text labels vertically aligned with the dashed lines written on semi-transparent background
    plt.text(0.00041, max(BS_CROSSECTION)/100000, '0.41 mm', rotation=90, verticalalignment='bottom', backgroundcolor='white')
    plt.text(0.001, max(BS_CROSSECTION)/100000, '1 mm', rotation=90, verticalalignment='bottom', backgroundcolor='white')
    plt.text(0.006, max(BS_CROSSECTION)/100000, '6 mm', rotation=90, verticalalignment='bottom', backgroundcolor='white')
    plt.xlabel('Bubble Radius (m)')
    plt.ylabel('Backscattering Cross-Section (m²)')
    plt.title('Backscattering Cross-Section vs Bubble Size at 38 kHz')
    plt.grid()
    plt.show()


    # --- Estimate flux --- #

    



    # Set the path to the echogram data
    #data_path = DATA_DIR / "CAGE_16_4_echograms"  #16-4 dataset
    # list the files in the directory
    #fileslist = os.listdir(data_path)
    #fileslist = [x for x in fileslist if x.endswith(".raw")]

    #pick one file to open
    #file_to_open = fileslist[0]
    #echodata = ep.open_raw(data_path / file_to_open, sonar_model="EK60")
    #calculate calibrated backscatter
    #echodata_cal = ep.calibrate.compute_Sv(echodata)
