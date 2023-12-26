import numpy as np
import pandas as pd

class GlobeTemperatureCalculator:
    """
    A class to calculate the globe temperature, which is a measure used to estimate thermal comfort or stress.
    It integrates the effects of air temperature, radiant heat, and wind speed.

    Parameters:
    S (float): Solar radiation received by the sensor (W/m2).
    f_db (float): Direct solar radiation fraction.
    f_dif (float): Diffuse solar radiation fraction.
    T_d (float): Dew point temperature. (Celsius)
    T_a (float): Actual air temperature. (Celsius)
    P (float): Atmospheric pressure (mb).
    h (float): Convective heat transfer coefficient.
    u (float): Wind speed in meters per second (m/s).
    z (float): Zenith angle in degrees. 
    B (float, optional): Pre-calculated constant B. If not provided, it will be calculated.
    C (float, optional): Pre-calculated constant C. If not provided, it will be calculated.
    """
    
    def __init__(self, S, f_db, f_dif, T_d, T_a, P, h, u, z, B=None, C=None):
        self.S = S
        self.f_db = f_db
        self.f_dif = f_dif
        self.T_d = T_d
        self.T_a = T_a
        self.P = P
        self.h = h
        self.u = u
        self.z = z
        self.B = B
        self.C = C
        self.stephan_boltzmann_constant = 5.67e-8  # Stefan-Boltzmann constant (W/m^2K^4)

    def calculate_e_a(self):
        """Calculate the atmospheric vapor pressure, e_a."""
        term1 = np.exp(17.67 * (self.T_d - self.T_a)/(self.T_d+243.5))
        term2 = 1.0007 + 0.00000346 * self.P
        term3 = 6.112 * np.exp((17.502 * self.T_a) / (240.97 + self.T_a))
        e_a = term1 * term2 * term3
        return e_a

    def calculate_epsilon_a(self):
        """Calculate the thermal emissivity, epsilon_a."""
        e_a = self.calculate_e_a()
        epsilon_a = 0.575 * e_a ** (1/7)
        return epsilon_a

    def calculate_B(self):
        """Calculate the constant B."""
        if self.B is not None:
            return self.B
        cos_z = np.cos(np.deg2rad(self.z))
        term1 = self.f_db / (4 * cos_z * self.stephan_boltzmann_constant)
        term2 = (1.2 * self.f_dif) / self.stephan_boltzmann_constant
        epsilon_a = self.calculate_epsilon_a()
        term3 = epsilon_a * self.T_a**4
        return (term1 + term2) * self.S + term3

    def calculate_C(self):
        """Calculate the constant C."""
        if self.C is not None:
            return self.C
        constant_factor = 5.3865 * 10**-8
        u = u * 3600.0 #convert to m/h from m/s
        return (self.h * self.u**0.58) / constant_factor

    def calculate_globe_temperature(self):
        """Calculate the globe temperature (T_g)."""
        B = self.calculate_B()
        C = self.calculate_C()
        upper = B + C * self.T_a + 7680000
        bottom = C + 256000
        T_g = upper / bottom
        return T_g