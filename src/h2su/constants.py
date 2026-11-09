"""Physical constants used across the package (SI unless stated)."""
from scipy import constants as _c

R_GAS = _c.gas_constant            # J/(mol K), CODATA via scipy.constants
F_PAPER = 96485.0                  # A s/mol, value printed in L26 (Rauls 2023) Sec 2.1
F_CODATA = _c.physical_constants["Faraday constant"][0]
M_H2 = 2.01588e-3                  # kg/mol, IUPAC standard atomic weight of H x 2
M_O2 = 31.998e-3                   # kg/mol, IUPAC standard atomic weight of O x 2
M_N2 = 28.0134e-3                  # kg/mol
P_STD = 101325.0                   # Pa
T_STD = 273.15                     # K
LHV_H2_KWH_PER_KG = 33.33          # kWh/kg (L4 uses 0.033 MWh/kg)
T0_K = 273.15
