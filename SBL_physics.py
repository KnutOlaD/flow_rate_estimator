'''
Package containing functions for physics calculations related to bubbles and surrounding water
for the nummerical bubble lab project.

Author: Knut Ola Dølven

Contents:

def get_viscosity(temperature,salinity): Returns the viscosity of water as a function of 
    temperature and salinity.
'''

import numpy as np
import scipy.io

def get_hydrostatic_pressure(depth,rho_sw=1025):
    '''
    Function that calculates the pressure at a given depth in seawater.'''

    #Constants
    g_acc = 9.81 #m/s^2 Gravitational acceleration
    #Calculate pressure
    pressure = rho_sw*g_acc*depth #Pa
    return pressure


def get_viscosity(temperature,salinity):
    '''
    Returns the viscosity of water as a function of temperature and salinity. 
    Calculates the seawater viscosity using the Sharqawy 2010 formulation.
    See documentation file for further details and theoretical background.
     
    Inputs:
    temperature: scalar or 1d array of temperature values in degrees Celsius

    salinity: scalar or 1d array of salinity values in g/kg
      
    Outputs:
    viscosity: scalar or 1d array of viscosity values in Pa s
    
    '''

    #check if the inputs are arrays or scalars and make them arrays if they are scalars
    if np.isscalar(temperature):
        temperature = np.array([temperature])
    if np.isscalar(salinity):
        salinity = np.array([salinity])
    
    #Convert temperature to Kelvin
    temperature_kelvin = temperature + 273.15 #K

    #Convert salinity from g/kg to mol/l
    salinity_molality = salinity/58.44 #mol/l, molar mass of NaCl is 58.44 g/mol
    salinity_SI = salinity*0.001 #kg/kg
    
    #Using IAPWS 2008 formulation for pure water viscosity
    mu_0 = 4.2844*10**-5 +(0.157*((temperature)+64.993)**2-91.296)**-1

    #Using Sharqawy 2010 formulation for seawatere  viscosity
    A_s = 1.541 + 1.998*10**-2*temperature - (9.52*10**-5)*temperature**2
    B_s = 7.974 - (7.561*10**-2)*temperature + (4.724*10**-4)*temperature**2
    
    #Alternate method, using Viswanath 2007 and Natarajan relation for pure water viscosity
    #A_mu_Vis = -4.5318
    #B_mu_Vis = -220.57
    #C_mu_Vis = 149.39

    #mu_0 = np.exp(A_mu_Vis + B_mu_Vis/(C_mu_Vis-temperature))

    #Constants for the viscosity of seawater - Kaminsky et al. 1981
    #A_s = 0.062 #A/(mol / l)(-1/2)Pa s
    #B_s = 0.0793 #A/(mol / l)(-1)Pa s
    #C_s = 0.008 #A/(mol / l)(-2)Pa s

    #Calculate the viscosity of pure water
    #mu_0 = A_mu*np.exp(B_mu/(temperature-C_mu))

    #Check if there are zeros in the salinity data, if so assume pure water
    if np.any(salinity==0):
        print('Zeros in salinity data, assume pure water')
        return mu_0
    else: #Calculate the viscosity of the saline water
        mu_s = mu_0*(1 + A_s*salinity_SI + B_s*salinity_SI**2)#Sharqawy 2010

    return mu_s
        
##########################################################################################

def get_density(temperature,salinity,pressure=None):
    '''
    Calculates the density of water using the UNESCO 1981 equation of state 
    neglecting the effect of pressure, i.e. simplified equation. Might add in pressure
    dependence later. 

    Inputs:
    temperature: scalar or 1d array of temperature values in degrees Celsius
    salinity : scalar or 1d array of salinity values in g/kg

    Outputs:
    density: scalar or 1d array of density values in kg/m^3
    '''
    
    #check if the inputs are arrays or scalars and make them arrays if they are scalars
    if np.isscalar(temperature):
        temperature = np.array([temperature])
    if np.isscalar(salinity):
        salinity = np.array([salinity])
    if np.isscalar(pressure):
        pressure = np.array([pressure])

    #Define coefficients
    a0 = 999.842594
    a1 = 6.793952*10**-2
    a2 = -9.095290*10**-3
    a3 = 1.001685*10**-4
    a4 = -1.120083*10**-6
    a5 = 6.536332*10**-9

    b0 = 8.24493*10**-1
    b1 = -4.0899*10**-3
    b2 = 7.6438*10**-5
    b3 = -8.2467*10**-7
    b4 = 5.3875*10**-9

    c0 = -5.72466*10**-3
    c1 = 1.0227*10**-4
    c2 = -1.6546*10**-6

    d0 = 4.8314*10**-4

    #Calculate density
    density = a0 + a1*temperature + a2*temperature**2 + \
        a3*temperature**3 + a4*temperature**4 + a5*temperature**5 + \
            (b0 + b1*temperature + b2*temperature**2 + b3*temperature**3 \
             + b4*temperature**4)*salinity + (c0 + c1*temperature + \
                c2*temperature**2)*salinity**1.5 + d0*salinity**2

    return density

##########################################################################################

def get_surf_tens(temperature=25,salinity=None):
    '''
    Function that calculates the surface tension of water [N/m] using the 
    Sharqawy et al. (2010) as a function of temperature (in degrees Celsius)
    Using Equation 27 for freswhater and Equation 28 for seawater. Can add more 
    and need to add theory in pdf. 
    Validity range is 0-40 C and 0-40 g/kg.
    
    Inputs:
    temperature: scalar or 1d array
        Temperature values in degrees Celsius
        Default is 25 degrees Celsius

    salinity: scalar or 1d array
        Salinity values in g/kg
        Default is None, which means that the surface tension of freshwater is 
        returned

    Outputs:
    surf_tens: scalar or 1d array of surface tension values in N/m

    '''

    #Check if the inputs are arrays or scalars and make them arrays if they are scalars
    if np.isscalar(temperature):
        temperature = np.array([temperature])
    if np.isscalar(salinity):
        salinity = np.array([salinity])

    #Fresh water
    surf_tens_fw = (0.2358*(1 - (temperature+273.15)/647.096)**1.256)*(1
                        -0.625*(1 - (temperature+273.15)/647.096))

    #Seawater
    if salinity != None:
        surf_tens = (1+(0.000226*temperature+0.00946)*np.log(
            1+0.0331*salinity))*surf_tens_fw
    else:
        surf_tens = surf_tens_fw

    return surf_tens

##########################################################################################

def get_bsd(model_choice='Gaussian',
            bsd_params=[0.0010,0.005,0,0.0020],
            resolution=100):
    """
    Function that returns the bubble size distribution which is needed
    to calculate the bubble rising speeds. The user can choose between
    different bubble size distribution models. The one used here is just a
    default is a simple Gaussian distribution centered at 10 mm and with a standard
    deviation of 5 mm and a range between 0 and 20 mm with 100 bins.
    Input is model_choice which designates the chosen bsd. List below shows options
    bsd_params which are the parameters for the chosen bsd model in array form 
    with the following information [mean, standard deviation, min bubble size, max bubble size]
    This function will allow for different bubble size distribution
    parametrizations. 

    Input:
    model_choice: string 
        Designates the chosen bubble size distribution model'
    bsd_params: list
        Parameters for the chosen bubble size distribution model in array form
        with the following information [mean, standard deviation, min bubble size, max bubble size]
    resolution: scalar
        Resolution of the bubble size distribution. Default is 100.

    Output:
    bsd: A 2xN bubble size distribution as a probability density function. The first column
    gives the probability of a bubble being of size of the corresponding row in the the 
    second column. Unit is meter.
    """
    #Bubble size bins:
    bsd = np.zeros((resolution,2))
    bsd[:,0] = np.linspace(bsd_params[2],bsd_params[3],resolution) 
    
    if model_choice == 'Gaussian':
        bsd[:,1] = (np.exp(-0.5*((bsd[:,0]-bsd_params[0]) #PUT IN THE BSD_PARAMS HERE. 
                      /bsd_params[1])**2))/(bsd_params[1]*np.sqrt(2*np.pi))
        #The bubble size distribution is normalized to 1 to get the probability density function

    #Normalize
    bsd[:,1] = bsd[:,1]/np.sum(bsd[:,1])

    return bsd

##########################################################################################

def get_brsm(bubble_size_range=[0.1*1e-3,6*1e-3],
             temperature=5, 
             salinity=35,
             model=None,
             tuning=None,
             resolution=100):
    '''
    Function that uses different bubble rising speed models to return the bubble rising 
    speed for all bubbles in the bubble size range with resolution resolution. 
    Supported models are listed below in the Inputs section.

    Inputs: #----------------------------------------------
    bubble_size_range: 1d-array 
        Array containing the minimum and maximum bubble size in m.
        Default is [0.1,2.5] m
    temperature: scalar 
        Gives temperature values in degrees Celsius.
        Default is 5 degrees Celsius
    salinity: scalar 
        Salinity in g/kg.
        Default is 35 g/kg
    model: string
        bubble rising speed model.
        Default is the model from Woolf (1993). 
    
        Alternatives are
    
        'Woolf1993' (Woolf, 1993). This is the default model. Linear increase in rising speed with 
        bubble size, then constant at 0.25m s^-1 for all bubbles larger a certain threshold size. 

        'Fan1990' (Fan and Tsuchiyua, 1990), has tuning input [c,d] which determines if the 
        model is for freshwater/saltwater and clean/dirty bubbles. The c parameter is set
        to 1.4 for seawater and 1.2 for pure freshwater. The d parameter is set to 1.6 for completely
        clean bubbles and 0.8 for dirty bubbles. I'm guessing there are valid intermediate
        values for these parameters as well, but see Fan & Tsuchiyua, 1990 and Leifer & Patro, 2002
        for details.

        'Woolf1991' (Woolf and Thorpe, 1991)

        'Leifer2002' (Leifer & Patro, 2002)

        'Veloso2015' (Leifer, 2000, clean bubbles)

    tuning: Some models comes with different parameter settings related to e.g. 
        clean/surfactant covered. See description of bubble rising speed models
        for details. This parameter can be various things.  
    
    resolution: scalar
        Number of points in the bubble size range to calculate the bubble rising speed for.    
        Default is 100 points.
        
    
    Outputs: #--------------------------------------------------
    brsm: 2d-array
        Array containing the bubble sizes in column 1 and the bubble rising speed for all bubbles 
        in the bubble size range

    #end
    '''

    ### Calculate input parameters
    viscosity_sw= get_viscosity(temperature,salinity) #Viscosity of the seawater [Pa s]
    density = get_density(temperature,salinity) #kg/m^3 #density of seawater [kg/m^3]
    #Would be better to use gsw, but using this function to avoid dependencies. 
    g_acc = 9.81 #m/s^2 #Gravitational acceleration [m/s^2]
    surf_tens = get_surf_tens(temperature,salinity) #N/m #Surface tension of seawater [N/m]
    visc_kinematic = viscosity_sw/density #m^2/s #Kinematic viscosity of seawater [m^2/s]
    #Recalculate the bubble size range to meters
    bubble_size_range = np.array(bubble_size_range) #[m]

    #Create the bubble array
    b_size_array = np.linspace(bubble_size_range[0],bubble_size_range[1],resolution) #[m]
    BRSM = np.zeros((resolution,2)) #Bubble rising speed matrix
    BRSM[:,0] = b_size_array #Bubble size array

    #Check what model to use
    if model == None:
        model = 'Woolf1993'
        print('No model chosen, using default model Woolf, 1993')

    if model == 'Woolf1993':
        BRSM[:,1] = 0.172*b_size_array**1.28*g_acc**0.76*visc_kinematic**-0.56
        #Set all rs higher than 0.25 m/s to 0.25 m/s
        BRSM[BRSM[:,1]>0.25,1] = 0.25
    elif model == 'Fan1990':
        if tuning == None:
            print('This model needs tuning parameters, specify tuning parameters [c,d],'
                  'see documentation for details. Using parameters for clean ' 
                  'bubbles in seawater as default.')
            c = 1.4
            d = 1.6
        else:
            c = tuning[0]
            d = tuning[1]
        #Calculate the Morton number
        Morton = (g_acc*viscosity_sw**4)/(density*surf_tens**3)
        BRSM[:,1] = (((density*g_acc*b_size_array**2)/
                        (3.68*(Morton**-0.038)*viscosity_sw))**-d+
                    ((c*surf_tens)/(density*b_size_array)+
                        g_acc*b_size_array)**(-d/2))**(-1/d)

    return BRSM

##########################################################################################

def get_rising_speed_pdf(bsd,brsm,resolution=100):
    '''
    Function that calculates the rising speed probability density function for a given
    bubble size distribution and bubble rising speed model.

    Inputs:
    bsd: 2d-array
        Bubble size distribution with bubble size in column 1 and probability of bubble
        size in column 2.
    BRSM: 2d-array
        Bubble rising speed model with bubble size in column 1 and bubble rising speed
        in column 2.
    resolution: scalar
        Number of points in the bubble size range to calculate the probability density
        for, i.e. the number of bins in the histogram.
    
    Outputs:
    RS_PDF: 2d-array
        Rising speed probability density function with rising speed in column 1 and 
        probability of rising speed in column 2.
    '''

    #Make bsd and BRSM the same size
    if len(bsd) != len(brsm):
        tmp = brsm
        brsm = np.zeros((len(bsd),2))
        print('Bubble size distribution and bubble rising speed model are not the' 
              'same size. Interpolating the bubble rising speed model array to match' 
              'the size of the bubble size distribution array.')
        brsm[:,1] = np.interp(bsd[:,0],tmp[:,0],tmp[:,1])
        brsm[:,0] = bsd[:,0]

    #Set bin edges for the histogram to accumulate probability for different rising speeds
    bin_edges = np.linspace(np.min(brsm[:,1]),np.max(brsm[:,1]),resolution)
    #Define the final rising speed probability density function 2d-array
    RS_PDF = np.zeros((len(bin_edges)-1,2))
    #Bin the rising speed in ascending order in bins spanning the range of the rising speed
    RS_PDF[:,1] = np.histogram(brsm[:,1],
                            resolution-1,
                            weights = bsd[:,1])[0]
    #Get the bin centers from the bin_edges in the histogram
    RS_PDF[:,0] = (np.linspace(np.mean(bin_edges[0:2]),
                                        np.mean(bin_edges[len(bin_edges)-1:len(bin_edges)]),
                                        len(bin_edges)-1))
    
    return np.array(RS_PDF)

def get_mass_from_volume(volume,temperature,pressure,gas='methane',method='ideal'):
    '''
    Function that calculates the number of moles of a volume of gas using either the ideal gas law or
    van der Waals equation of state. The function can be used to calculate the mass of a gas
    volume at a given depth using a single value or a list of values for volume, density and depth.

    van der Waals equation
    (P+a*n^2/V^2)(V-n*b)=n*R*T  --> n^3(-a*b/V^2)+n^2(a/V)-n(P*b+R*T)+P*V = 0
    where a and b are constants for the gas type, V is the volume, n is the number of moles, R is
    the gas constant, T is the temperature and P is the pressure.

    Inputs: 
    volume: (N,)array_like or scalar
       Volume of gas [m3]
    temperature: (N,)array_like or scalar
         temperature of gas (degC)
    pressure: (N,)array_like or scalar
        pressure [Pa]
    gas: string
        Gas type, default is methane, supported types are 'methane', 'carbondioxide', 'nitrogen',
        'argon' and 'oxygen'
    method: string
        Method for calculating molar mass, options are
          'ideal' - Ideal gas law
          'vdw' - van der Waals equation of state
          'Mario' - Mario's equation of state
        Default is 'ideal'
    
    Outputs:
    moles: (N,)array_like or scalar
        Number of moles of gas [mol]
    '''
    #Check if inputs are arrays or scalars
    if isinstance(volume,(list,np.ndarray)):
        volume = np.array(volume)
        temperature = np.array(temperature)
        pressure = np.array(pressure)
    else:
        volume = np.array([volume])
        temperature = np.array([temperature])
        pressure = np.array([pressure])

    #Check if inputs are of same length
    if not (len(volume) == len(temperature) == len(pressure)):
        raise ValueError('Inputs must be of same length')
    
    ### Set input parameters ###

    #Get temperature to Kelvin
    temperature = 273.15 + temperature

    #Check what gas it is and get the correct gas constants
    if gas == 'methane':
        molar_mass = 16.04 #g/mol
        a = 2.283*10**(-1) #Pa*m^6/mol^2
        b = 4.301*10**(-5) #m^3/mol
        R = 8.314 #J/mol*K
    elif gas == 'carbondioxide':
        molar_mass = 44.01
        raise ValueError('Gas not supported yet')
    elif gas == 'nitrogen':
        molar_mass = 28.01
        raise ValueError('Gas not supported yet')
    elif gas == 'argon':
        molar_mass = 39.95
        raise ValueError('Gas not supported yet')
    elif gas == 'oxygen':
        molar_mass = 32.00
        raise ValueError('Gas not supported yet')
    
    #Set up equation
    if method == 'ideal':
        #Ideal gas law
        moles = pressure*volume/(R*temperature)
        coeff = []
    elif method == 'vdw':
        #van der Waals equation of state
        #Solve the cubic equation of state for n
        #Coefficients for cubic equation. 
        #Flatten if length of all the inputs are 1
        if len(volume) == 1 and len(temperature) == 1 and len(pressure) == 1:
            volume = volume.flatten()
            temperature = temperature.flatten()
            pressure = pressure.flatten()
        
        coeff = np.array([-a*b/volume**2,a/volume,-(pressure*b+R*temperature),pressure*volume])
        #Find out if there's more than 1 equation
        if len(coeff.shape) > 1:
            #Then loop over and find all the roots
            moles = np.zeros(len(coeff[0,:]))
            for i in range(len(coeff[0,:])):
                #Solve the cubic equation and extract real root of the third root
                moles[i] = np.real(np.roots(coeff[:,i])[2].astype(float))
        else:
            #Solve the cubic equation and extract real root of the third root
            moles = np.real(np.roots(coeff[:,0])[2].astype(float))
            #Calculate mass from n
            #mass = n*molar_mass
    elif method == 'Mario':
        #Calculate the density of the gas at the given temperature and pressure
        density = pressure*molar_mass/(R*temperature)
        #Dont understand this.. 
    elif method != 'ideal' and method != 'vdw' and method != 'Mario':
        raise ValueError('Method not recognized. Choose either "ideal", "vdw" or "Mario"')
    return(moles)

def get_gas_density(temperature,pressure,gas_type='methane',method = 'vdw'):
    '''
    Function that calculates the gas density of a given gas at a given temperature
    and pressure using the volume_to_mass function to calculate the number of moles of gas
    and then calculates the density from that. Can use both ideal gas law and van der Waals

    Inputs:
    temperature: scalar or 1d-array
        Temperature in degrees Celsius
    pressure: scalar or 1d-array
        Pressure in Pa
    gas_type: string
        Type of gas. Default is 'CH4' for methane. Other options can be added later.

    Outputs:
    gas_density: scalar or 1d-array
        Gas density in kg/m^3
    '''

    #Check if the inputs are arrays or scalars and make them arrays if they are scalars
    if np.isscalar(temperature):
        temperature = np.array([temperature])
    if np.isscalar(pressure):
        pressure = np.array([pressure])
    #Convert temperature to Kelvin
    temperature_kelvin = temperature + 273.15 #K
    #Molar mass of gases in kg/mol
    molar_masses = {
        'methane': 0.01604, #Methane
        #Add more gases here as needed
    }
    if gas_type not in molar_masses:
        raise ValueError(f"Gas type '{gas_type}' not recognized.")
    M = molar_masses[gas_type] #kg/mol
    # Calculate the number of moles per cubic meter using the get_mass_from_volume_vdW function
    kg_per_m3 = get_mass_from_volume(volume=1, temperature=temperature_kelvin, pressure=pressure, gas=gas_type, method=method) * M
    return kg_per_m3

def get_sound_speed(temperature,salinity,depth):
    '''
    Function that calculates the sound velocity in seawater using the Mackenzie 1981
    equation for sound velocity in seawater.

    Inputs:
    temperature: scalar or 1d-array
        Temperature in degrees Celsius
    salinity: scalar or 1d-array
        Salinity in g/kg
    depth: scalar or 1d-array
        Depth in meters

    Outputs:
    sound_velocity: scalar or 1d-array
        Sound velocity in m/s
    '''

    #Check if the inputs are arrays or scalars and make them arrays if they are scalars
    if np.isscalar(temperature):
        temperature = np.array([temperature])
    if np.isscalar(salinity):
        salinity = np.array([salinity])
    if np.isscalar(depth):
        depth = np.array([depth])

    #Calculate sound velocity using Mackenzie 1981 equation
    sound_velocity = 1448.96 + 4.591*temperature - 5.304*10**-2*temperature**2 + \
        2.374*10**-4*temperature**3 + 1.340*(salinity - 35) + 1.630*10**-2*depth + \
            1.675*10**-7*depth**2 - 1.025*10**-2*temperature*(salinity - 35) - \
                7.139*10**-13*temperature*depth**3
    return sound_velocity


def generate_bsd_pdf(BSD,min,max,vectorsize):
    '''
    Generate a probability density function (PDF) for bubble size distribution (BSD)
    for a given bubble plume. Does this using a polynomial fit of order 10
    
    Inputs:
    BSD: 2D array with shape(n_observations, 2) where column 0 is
    the bubble sizes and column 1 is the corresponding observations
    min: minimum bubble size to consider (mm)
    max: maximum bubble size to consider (mm)
    vectorsize: number of bins in the PDF

    Outputs:
    pdf: probability density function as a 1D array of size vectorsize
    bin_centers: centers of the bins corresponding to the PDF
    '''

    bsd_pdf = np.zeros((vectorsize,2))

    # Generate bubble sizes based on the polynomial fit
    coeffs = np.polyfit(BSD[:,0], BSD[:,1], 10)
    bubble_sizes = np.linspace(np.min(BSD[:,0]), np.max(BSD[:,0]), 100)
    bubble_observations = np.polyval(coeffs, bubble_sizes)

    #Determine the number of bins
    bins = np.linspace(min, max, vectorsize+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Generate bubble sizes based on the polynomial fit
    bubble_sizes = np.linspace(np.min(BSD[:,0]), np.max(BSD[:,0]), 100)
    bubble_observations = np.polyval(coeffs, bubble_sizes)

    #ensure no negative values
    bubble_observations[bubble_observations < 0] = 0

    pdf = bubble_observations / np.sum(bubble_observations)

    bsd_pdf[:,1] = bin_centers
    bsd_pdf[:,0] = np.interp(bin_centers, bubble_sizes, pdf)


    return np.array(bsd_pdf)

def get_methane_properties(temperature=298.15,
                           pressure=101325,
                           property='list',
                           return_all=False):
    '''
    Function that returns the properties of methane gas at given temperature and pressure.
    Inputs:
    temperature: scalar or 1d-array
        Temperature in degrees Celsius
    pressure: scalar or 1d-array
        Pressure in Pa
    property: string
        Property to return. Included properties can be displayed by calling the function with
        property='list' (this is also the default). Supported properties are
        'density', 'dynamic viscosity', 'kinematic viscosity', 'specific heat at constant pressure',
        'specific heat at constant volume', 'thermal conductivity', 'thermal diffusivity', 'polytropic index'
    return_all: boolean
        If True, returns all properties as a dictionary. Default is False.
    Outputs:
    prop_value: scalar or 1d-array
        Value of the requested property
    '''

    #List of supported properties, have substituted in Mario's numbers for methane at room temperature and pressure
    supported_properties = {
        'density': 0.656, #kg/m^3
        'dynamic viscosity': 1.1e-5, #Pa s
        'kinematic viscosity': 1.676e-5, #m^2/s
        'bulk viscosity': 1.33*1.1e-5, #Pa s, parametrization from Prangsma (1972)
        'specific heat at constant pressure': 2220, #J/kg K
        'specific heat at constant volume': 1690, #J/kg K
        'thermal conductivity': 0.034, #W/m K
        'thermal diffusivity': 9.19*10**-7,#NON MARIO VALUE: 0.00023346, #m^2/s
        'polytropic index': 1.31 #dimensionless
    }

    #supported_properties['thermal diffusivity'] = supported_properties['thermal conductivity'] / (supported_properties['density'] * supported_properties['specific heat at constant pressure'])

    if property == 'list':
        return list(supported_properties.keys())
    elif return_all:
        return supported_properties
    elif property in supported_properties:
        return supported_properties[property]
    else:
        raise ValueError(f"Property '{property}' not recognized. Supported properties are: {list(supported_properties.keys())}")

def get_gaussian_distribution(mean, std, min, max, vectorsize):
    '''
    Generate a Gaussian probability distribution
    
    Inputs:
    mean: mean of the Gaussian distribution (mm)
    std: standard deviation of the Gaussian distribution (mm)
    min: minimum size to consider (mm)
    max: maximum size to consider (mm)
    vectorsize: number of bins in the PDF

    Outputs:
    pdf: probability density function as a 1D array of size vectorsize
    bin_centers: centers of the bins corresponding to the PDF
    '''

    # Determine the number of bins
    bins = np.linspace(min, max, vectorsize+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Generate Gaussian distribution
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bin_centers - mean) / std) ** 2)

    # Normalize the PDF
    pdf = pdf / np.sum(pdf)

    return pdf, bin_centers

def get_Minnaert_frequency(radius_bubble=0.0048,
                           pressure_static=50000,
                           specific_heat_ratio=None,
                            density_water=1025):
    '''
    Calculates the Minnaert frequency for a bubble with given radius, gas content, and surrounding fluid properties.
    Inputs:
    radius_bubble: scalar
        Bubble radius in meters
    pressure_hydrostatic: scalar
        Hydrostatic pressure in Pa
    specific_heat_ratio: scalar
        Specific heat ratio of the gas inside the bubble. Uses specific heat for methane if None.
        Note: Specific heat ratio is specific heat at constant pressure divided by specific heat at constant volume.
    density_water: scalar
        Density of the surrounding water in kg/m^3
    Outputs:
    freq_Minnaert: scalar
        Minnaert frequency in rad/s. This is the natural frequency of oscillation for the bubble.

    '''
    #substituted in Mario's number for specific heat ratio. 
    if specific_heat_ratio == None:
        specific_heat_ratio = 1.4 #Old value: 1.31 #Specific heat ratio for methane (Cp = 35.8 J/mol K, Cv = 27.4 J/mol K)

    freq_Minnaert = np.sqrt((3*specific_heat_ratio*pressure_static)/(density_water*radius_bubble**2))
                    
    return freq_Minnaert   


def get_re_radiation_damping_coefficient(radius_bubble=0.0048,
                                         freq_Minnaert=33000,
                                         speed_sound=1500):
    '''
    Calculates the dimensionless re-radiation damping coefficient for bubble backscattering.

    Inputs:
    freq_Minnaert: scalar
        Minnaert frequency in Hz
    radius_bubble: scalar
        Bubble radius in meters
    speed_sound: scalar
        Sound speed in m/s

    Outputs:
    delta_re_rad: scalar
        Dimensionless re-radiation damping coefficient
    '''

    delta_re_rad = (freq_Minnaert*radius_bubble)/speed_sound
    return delta_re_rad


def get_thermal_damping_coefficient(radius_bubble=0.0048,
                                    freq_Minnaert=33000,
                                    diffusivity_thermal=0.00023346, #Diffusivity of heat of the gas [m^2/s] FIX
                                    specific_heat_ratio=1.31,
                                    ):

    '''
    Calculates the dimensionless thermal damping coefficient a bubble with radius r.

    Inputs:
    radius_bubble: scalar
        Bubble radius in meters
    freq_Minnaert: scalar
        Minnaert frequency in Hz
    diffusivity_thermal: scalar
        Thermal diffusivity of the surrounding fluid in m^2/s
    specific_heat_ratio: scalar
        Specific heat ratio of the gas inside the bubble. Default is for methane.

    Outputs:
    delta_thermal: scalar
        Dimensionless thermal damping coefficient
    '''

    delta_thermal = (3*(specific_heat_ratio-1)/(radius_bubble))*np.sqrt((diffusivity_thermal)/(2*freq_Minnaert))

    return delta_thermal
#Using Mario's numbers now
def get_viscous_damping_coefficient(radius_bubble=0.0048,
                                    freq_Minnaert=33000,
                                    viscosity_shear=1.519*10**-3,#viscosity_shear=0.01107*0.001, #Shear viscosity for the gas. This is dynamic shear viscosity for methane [Pa s] (0.01107 cP at 25 C)
                                    viscosity_bulk = 2.2*1.519*10**-3,#viscosity_bulk=1.3*0.01107*0.001, #Bulk viscosity for the gas. This is also called volume viscosity [Pa s]
                                    density_water=1025):
    '''
    Calculates the dimensionless viscous damping coefficient for bubble backscattering.
    Inputs:
    radius_bubble: scalar
        Bubble radius in meters
    freq_Minnaert: scalar
        Minnaert frequency in Hz
    viscosity_shear: scalar
        Shear viscosity of the gas in Pa s
    viscosity_bulk: scalar
        Bulk viscosity of the gas in Pa s, using Prangsma (1972) parametrization at 293 K (should be at least temp dependent)
    density_sea_water: scalar
        Density of sea water in kg/m^3 (should be temp and salinity dependent)
    Outputs:
    delta_viscous: scalar
        Dimensionless viscous damping coefficient  

    '''
    density_viscous = (viscosity_shear + 3*viscosity_bulk)/(density_water*freq_Minnaert*radius_bubble**2)
    return density_viscous


def get_delta_damping_coefficient(radius_bubble=0.0048,
                                      freq_Minnaert=10000000,
                                      diffusivity_thermal=9.19*10**-7,#old value:0.00023346,
                                      viscosity_shear=1.519*10**-3,#old value:0.01107*0.001,
                                      viscosity_bulk=2.2*1.519*10**-3,#old value:1.3*0.01107*0.001,
                                      density_water=1025,
                                      speed_sound=1500,
                                      specific_heat_ratio=1.4#old value:1.31
):
    '''
    Calculates the dimensionless damping coefficient for bubble backscattering.
    This is simply the sum of the three damping coefficients, but doing it like this to keep
    the code modular and easy to read.

    Inputs:
    radius_bubble: scalar
        Bubble radius in meters
    freq_Minnaert: scalar
        Minnaert frequency in Hz
    diffusivity_thermal: scalar
        Thermal diffusivity of the surrounding fluid in m^2/s
    viscosity_shear: scalar
        Shear viscosity of the gas in Pa s
    viscosity_bulk: scalar
        Bulk viscosity of the gas in Pa s
    density_sea_water: scalar
        Density of sea water in kg/m^3
    speed_sound: scalar
        Sound speed in m/s
    specific_heat_ratio: scalar
        Specific heat ratio of the gas inside the bubble. Default is for methane.

    Outputs:
    delta_damping: scalar
        Dimensionless damping coefficient
    '''

    delta_damping = get_re_radiation_damping_coefficient(radius_bubble=radius_bubble,
                                                           freq_Minnaert=freq_Minnaert,
                                                           speed_sound=speed_sound) + \
                    get_thermal_damping_coefficient(radius_bubble=radius_bubble,
                                                           freq_Minnaert=freq_Minnaert,
                                                           diffusivity_thermal=diffusivity_thermal,
                                                           specific_heat_ratio=specific_heat_ratio) + \
                    get_viscous_damping_coefficient(radius_bubble=radius_bubble,
                                                       freq_Minnaert=freq_Minnaert,
                                                       viscosity_shear=viscosity_shear,
                                                       viscosity_bulk=viscosity_bulk,
                                                       density_water=density_water)
    return delta_damping

def get_resonant_bubble_size(freq_echosounder=38000,
                             pressure_static=50000,
                             density_water=1025,
                             specific_heat_ratio=1.31):
    
    '''
    Calculates the resonant bubble size for a given echosounder frequency using the Minnaert frequency.
    Inputs:
    freq_echosounder: scalar
        Frequency in Hz of the echosounder
    pressure_hydrostatic: scalar
        Hydrostatic pressure in Pa
    density_water: scalar
        Density of the surrounding water in kg/m^3
    specific_heat_ratio: scalar
        Specific heat ratio of the gas inside the bubble. 
    
    Outputs:
    radius_resonant_bubble: scalar
        Resonant bubble radius in meters
    '''
    radius_resonant_bubble = (1/(2*np.pi*freq_echosounder)) * np.sqrt((3*specific_heat_ratio*pressure_static)/(density_water))
    return radius_resonant_bubble

def get_bubble_bs_oscillation(radius_bubble,
                                          freq_echosounder=38000,
                                          speed_sound=1500):
    '''
    Estimates the oscillating part of the product in Thuyraisingham (1997) equation for bubble backscattering cross-section.

    Inputs:
    radius_bubble: scalar
        bubble radius in meters
    frequency: scalar
        Frequency in Hz of the echosounder

    Outputs:
    oscillation_part: scalar
        Oscillating part of the bubble backscattering cross-section equation
    '''
    #Calculate the wave number
    wave_number = 2*np.pi*freq_echosounder/speed_sound  # in 1/m

    #Calculate the oscillating part of the bubble backscattering cross-section equation
    oscillation_part = ((np.sin(wave_number*radius_bubble)/(wave_number*radius_bubble))**2)/(
        1+(wave_number*radius_bubble)**2)

    return oscillation_part

def get_bubble_bs_damping(radius_bubble,
                                      radius_resonant_bubble=0.0048,
                                        delta_coeff=0.04):
    
    '''
    Estimates the damping part of the product in Thuyraisingham (1997) equation for bubble backscattering cross-section.
    Inputs:
    radius_bubble: scalar
        bubble radius in meters
    radius_resonant_bubble: scalar
        Resonant bubble radius in meters
    delta_coeff: Scalar
        Dimensionless damping coefficient. Calculated using the damping function
    Outputs:
    damping_part: scalar
        Damping part of the bubble backscattering cross-section equation
    '''
    
    damping_part = (radius_bubble**2)/(
        (((radius_resonant_bubble/radius_bubble)**2)-1)**2+(
            delta_coeff**2))

    return damping_part

def get_bubble_backscattering_crosssection(radius_bubble,
                                           radius_resonant_bubble=0.0048,
                                           freq_echosounder=38000,
                                           delta_coeff=0.04,
                                           speed_sound=1500):
    '''
    Estimates bubble backscattering cross-section for a given bubble size distribution 
    assuming spherical bubbles using the equation from Thuraisingham (1997).

    Inputs:
    radius_bubble: scalar
        bubble radius in meters
    radius_resonant_bubble: scalar
        Resonant bubble radius in meters
    frequency: scalar
        Frequency in Hz of the echosounder
    delta_coeff: Scalar
        Dimensionless damping coefficient. Calculated using the damping function
    speed_sound: scalar
        Sound speed in m/s
    Outputs:
    backscattering_crosssection: 1D array
        Array of backscattering cross-sections for each bubble size in m^2
    '''

    #Calculate the wave number
    #wave_number = 2 * np.pi * freq_echosounder / speed_sound  # in rad/m OLD
    #The wave number must have unit of 1/m otherwise the units in the formula will be wrong
    wave_number = 2*np.pi* freq_echosounder/speed_sound

    #Calculate the backscattering cross-section using Thuraisingham (1997) equation
    backscattering_crosssection = 4*np.pi*((radius_bubble**2)/(
        (((radius_resonant_bubble/radius_bubble)**2)-1)**2+(
            delta_coeff**2)))*(
            ((np.sin(wave_number*radius_bubble)/(wave_number*radius_bubble))**2)/(
                1+(wave_number*radius_bubble)**2)
            )

    return backscattering_crosssection



def get_flux_amplifier_PSI(radius_bubble,
                           bubble_size_PDF,
                           backscattering_crossection_array,
                           rise_velocity_array,
                           speed_sound=1500,
                           transmit_pulse_length=0.001):
    '''
    Function that calculates the flux amplifier PSI for a given bubble size distribution,
    backscattering cross-section and rise velocity.

    Inputs:
    radius_bubble: 1D array
        Array of bubble radii in meters
    bubble_size_PDF: 1D array
        Probability density function of bubble sizes
    backscattering_crossection_array: 1D array
        Array of backscattering cross-sections for each bubble size in m^2
    rise_velocity_array: 1D array
        Array of rise velocities for each bubble size in m/s
    speed_sound: scalar
        Sound speed in m/s
    transmit_pulse_length: scalar
        Transmit pulse length in seconds for the echosounder

    Outputs:
    PSI: scalar
        Flux amplifier PSI
   
    '''
    #Calculate the integrand for PSI
    volume_coefficient = (8*np.pi)/(3*speed_sound*transmit_pulse_length)
    #calclate the integral of the denominator 
    integral_numerator = np.sum(radius_bubble**3 * bubble_size_PDF * rise_velocity_array)
    #calculate the integral of the numerator
    integral_denominator = np.sum(backscattering_crossection_array * bubble_size_PDF)
    #sum
    PSI = volume_coefficient * (integral_numerator / integral_denominator)
    return PSI[0]

def get_mass_flux(flux_amplifier_PSI,
                  density_gas = 1, #if one wants volume flux instead of mass flux, set density_gas to 1 kg/m^3
                  target_strength_average_dB=-70 #average target strength in dB of insonified targets
                  ): 
    '''
    Function that calculates the mass flux (or volume flux if you set gas density to 1) of gas from the 
    flux amplifier PSI and average target strength of the insonified bubbles.
    Inputs:
    flux_amplifier_PSI: scalar
        Flux amplifier PSI
    density_gas: scalar
        Density of the gas in kg/m^3. Default is 1 kg/m^3, which gives volume flux instead of mass flux.
    target_strength_average_dB: scalar
        Average target strength in dB of the insonified bubbles.
    Outputs:
    mass_flux: scalar
        Mass flux of gas in kg/m^2/s
    '''
    mass_flux = density_gas * flux_amplifier_PSI * 10**(target_strength_average_dB/10)
    return mass_flux

#################################################
################# TESTING #######################
#################################################


if __name__ == '__main__':
    testing = False
    if testing == True:
        print('Running tests for SBL_physics.py')
    else:
        print('This is a module with functions for calculating various physical properties related to seawater and bubbles. Contains the functions:' \
        '' \
        '\nget_viscosity(temperature,salinity)' \
        '\nget_density(temperature,salinity)' \
        '\nget_surf_tens(temperature,salinity)' \
        '\nget_bsd(model_choice,bsd_params,resolution)' \
        '\nget_brsm(bubble_size_range,temperature,salinity,model,tuning,resolution)' \
        '\nget_rising_speed_pdf(bsd,brsm,resolution)' \
        '\nget_mass_from_volume(volume,temperature,pressure,gas,method)' \
        '\nget_gas_density(temperature,pressure,gas_type,method)' \
        '\nget_sound_speed(temperature,salinity,depth)' \
        '\nget_methane_properties(temperature,pressure,property,return_all)' \
        '\ngenerate_bsd_pdf(BSD,min,max,vectorsize)'
        '\nget_gaussian_distribution(mean, std, min, max, vectorsize)'
        '\nget_Minnaert_frequency(radius_bubble,pressure_static,specific_heat_ratio,density_water)'
        '\nget_re_radiation_damping_coefficient(radius_bubble,freq_Minnaert,speed_sound)'
        '\nget_thermal_damping_coefficient(radius_bubble,freq_Minnaert,diffusivity_thermal,specific_heat_ratio)'
        '\nget_viscous_damping_coefficient(radius_bubble,freq_Minnaert,viscosity_shear,viscosity_bulk,density_water)'
        '\nget_delta_damping_coefficient(radius_bubble,freq_Minnaert,diffusivity_thermal,viscosity_shear,viscosity_bulk,density_water)'
        '\nget_resonant_bubble_size(freq_echosounder,pressure_static,density_water,specific_heat_ratio)'
        '\nget_bubble_backscattering_crosssection(radius_bubble,radius_resonant_bubble,frequency,delta_coeff,speed_sound)' \
        '' \
        'Currently running no tests.')
    #Test get_viscosity
##########################################################################################