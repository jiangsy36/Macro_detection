"""
calculation of the constraints on dark matter from white dwarfs and neutron star
"""

import numpy as np
from scipy import integrate
from scipy import interpolate
from time import time
import matplotlib.pyplot as plt
import os
from sympy import symbols, solve
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

GCF = 6.70883e-39  # Gravitational constant in GeV^-2
mPL = GCF ** -0.5  # Planck mass in GeV

GeV_in_g = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm = 3.085677581e24  # 1 Mpc in cm
kpc_to_cm = 1e-3*Mpc_in_cm  # 1 kpc in cm
pc = 1e-8*kpc_to_cm #1pc in km
c = 299792.458  # in km/s
cm_in_invkeV = 5.067730938543699e7  # 1 cm in keV^-1
cm_in_invGeV = cm_in_invkeV*1e6
year_in_s = 3.168808781402895e-8  # 1 year in s
GeV_in_invs = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

Msun = 1.989e33  #g
year_in_s = 365 * 24 * 3600  # 1 year in s
Gyr_in_s = 1e9 * year_in_s

alpha_EM = 1/137



colors = {
    'LISA': ('#FF4444', (1.0, 0.3, 0.3, 0.7)),    
    'TianQin': ('#4444FF', (0.3, 0.3, 1.0, 0.7)), 
    'Taiji': ('#AA44AA', (0.7, 0.3, 0.7, 0.7))   
}



detector_alphas = {
    'TianQin': 0.25,
    'LISA': 0.2, 
    'Taiji': 0.15
}



class WD_constraints:

	def __init__(self):
		data_path = os.path.dirname(__file__)
		if data_path == "":
			WD_filename = "MWDD-export.csv"
		else:
			WD_filename = data_path + "/MWDD-export.csv"

		self.det_names, *self.det_WD_data = np.genfromtxt(WD_filename, delimiter=",", unpack=True, dtype=None,
														  encoding="utf-8", skip_header=1)
		self.det_params = ("mass", "age")
		self.det = dict(zip(self.det_params, self.det_WD_data))

	def find_and_remove_missing_values(self):
		"""
		Find missing values in 'mass' and 'age' columns, display corresponding 'det_names',
		and remove rows with missing values or invalid mass values.
		"""
		# Convert data to NumPy arrays for easier handling
		det_names_array = np.array(self.det_names)
		mass_array = np.array(self.det["mass"])
		age_array = np.array(self.det["age"])

		# Check for missing values (e.g., NaN)
		missing_mask = np.isnan(mass_array) | np.isnan(age_array)

		# Check for invalid mass values (mass < 0.8 or mass > 1.4)
		invalid_mass_mask = (mass_array < 0.8) | (mass_array > 1.4)

		# Combine both masks
		invalid_mask = missing_mask | invalid_mass_mask

		# Display detector names that will be removed
		if np.any(invalid_mask):
			print("Rows with missing or invalid values:")
			print(det_names_array[invalid_mask])

		# Remove rows with invalid values
		self.det_names = det_names_array[~invalid_mask]
		self.det["mass"] = mass_array[~invalid_mask]
		self.det["age"] = age_array[~invalid_mask]

		print(len(self.det["mass"]))

	def plot_range(self):
		fig, ax = plt.subplots(figsize=(10, 5))
		transparency = 0.4

		#print(len(self.det["mass"]))
		ax.plot(self.det["mass"], self.det["age"], 'o', ms=10, alpha=transparency)

		ax.set_xscale('log')
		ax.set_xlabel(R'$mass$')
		ax.set_ylabel(R'$age$')
		ax.set_yscale('log')
		ax.grid()
		#ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
		plt.tight_layout()
		plt.show()


class DW_analysis(WD_constraints):

	def __init__(self, verbose=False, foldername="", overview_title=""):
		if verbose:
			print("\n")
			print("   ----------------------------")
			print("       DW Spectrum Analysis    ")
			print("   ----------------------------")
			print("\n")

		super(DW_analysis, self).__init__()
		self.verbose = verbose
		self.overview_title = overview_title
		self.foldername = foldername

		self.lam1 = 1.3 * 1e-4  #cm
		self.lam2 = 2.5 * 1e-5  #cm
		self.rho1 = 2e8  #g/cm^3
		self.rho2 = 1e10  #g/cm^3
		self.rsolar = 8.3  #solar position kpc
		self.rho_DM = 0.4  #DM energy density GeV/cm^3
		self.v_DM = 3 * 1e7  #DM velocity cm/s

		self.find_and_remove_missing_values()


	def rho_WD(self, MWD):
		"""
		Eq.1 of 1911.08883
		MWD: mass of white dwarfs unit:Msun
		:return: density of white dwarfs unit:g/cm^3
		"""

		c0, c1, c2, c3, c4, c5, c6 = 1.003, -0.309, -1.165, 2.021, -2.060, 1.169, -0.281
		term1 = c0 * MWD ** 0 + c1 * MWD ** 1 + c2 * MWD ** 2 + c3 * MWD ** 3 + c4 * MWD ** 4 + c5 * MWD ** 5 + c6 * MWD ** 6

		return 1.95 * 1e6 * (term1 ** (-2) - 1) ** (3 / 2)

	def compute_all_densities(self):
		"""
		Compute densities for all white dwarfs using the loaded mass data.
		:return: List of densities
		"""
		densities = []
		for mass in self.det["mass"]:
			try:
				density = self.rho_WD(mass)
				densities.append(density)
			except ValueError as e:
				print(f"Error computing density for mass {mass}: {e}")
				densities.append(None)  # Append None for invalid values
		return densities

	def Gamma_meet(self, MWD, MDM, alphat, fx=0.01):
		"""
		encounter rate
		MDM in unit of g
		unit Gyr^-1
		"""
		rhoWD = self.rho_WD(MWD)  #g/cm^3

		MWD_in_g = MWD * Msun #g
		MWD_in_GeV = MWD * Msun / GeV_in_g  #GeV
		MDM_in_GeV = MDM / GeV_in_g  # GeV

		RWD = (MWD_in_g / rhoWD) ** (1 / 3) * (3 / (4 * np.pi)) ** (1 / 3)  #cm
		RWD_in_GeV = RWD * cm_in_invGeV  #GeV^-1
		v_esc_WD = np.sqrt(2 * GCF * MWD_in_GeV / RWD_in_GeV) #dimensionless
		v_DM = self.v_DM * 1e-5 / c #dimensionless
		
		lam = 1e-1*pc*1e5
		x = np.log(alphat/(2*lam*v_DM**2/v_esc_WD**2/RWD+1/np.log(alphat)))
		bcl_max = np.minimum(RWD/v_DM * np.sqrt((v_esc_WD ** 2*(1+alphat)+v_esc_WD ** 4*alphat**2/4) / (1-v_esc_WD ** 2)), np.sqrt(lam*x*(v_esc_WD**2*RWD/v_DM**2+lam*x)))
		Gamma = self.rho_DM / MDM_in_GeV * self.v_DM * np.pi * bcl_max ** 2 #s^-1

		return fx* Gamma * Gyr_in_s

	def R_env(self, MWD, age):
		"""
		width of a non-degenerate white dwarf envelope 
		"""
		X = 0
		Z = 1
		mu = 1/0.5729
		mu_e = 2
		T = 5.51*1e18*Z*(1+X)*(mu_e/mu)*(mu_e/mu-1)*(1/age)
		T_star = T**(1/2.5) #unit:K
		k = 1.38*1e-23 #J/K
		mu_a = 1.66*1e-27 #atomic mass unit, kg
		G = 6.67*1e-11 #Newton constant, N m^2/kg^2
		rhoWD = self.rho_WD(MWD)  #g/cm^3
		MWD_in_g = MWD * Msun  #g
		MWD_in_kg = MWD_in_g * 1e-3 #kg
		RWD_in_m = (MWD_in_g / rhoWD) ** (1 / 3) * (3 / (4 * np.pi)) ** (1 / 3) *1e-2 #unit:m
		x = 4.25*T_star*(k/mu/mu_a)*(RWD_in_m/G/MWD_in_kg)+1
		Renv = RWD_in_m*(1-1/x) #unit:m

		return Renv

	def env_condition(self, MWD, age):
		"""
		unit:cm^2/g
		"""
		rhoWD = self.rho_WD(MWD)*1e-3  #g/cm^3
		#rhoWD = 1e3
		Renv_in_cm = self.R_env(MWD, age)*1e2 #cm
		cond = 1/(rhoWD*Renv_in_cm)

		return cond

	def compute_all_env(self):
		"""
		Compute sigma/M_X by considering all valid white dwarfs
		"""
		valid_WDs= []  
		conds = [] 
		for name, mass, age in zip(self.det_names, self.det["mass"], self.det["age"]):
			try:
				cond = self.env_condition(mass, age)
				if cond:  # If a valid solution exists
					valid_WDs.append(name)  # Add detector name
					conds.append(cond)  # Add valid M_DM_max value
			except Exception as e:
				print(f"Error computing cond for {name}: {e}")
		return conds



	def M_DM_max(self, MWD, age, alphat):
		"""
		Eq.19 of 2306.14981, but we replace 2.3 with 3
		"""
		x = symbols('x')
		equation = self.Gamma_meet(MWD, x, alphat) * age - 3
		sol = solve(equation, x)

		return sol

	def compute_all_M_DM_max(self, alphat):
		"""
		Compute M_DM_max (the maximum value of DM mass from the white dwarf constraints) for all white dwarfs in the dataset.
		:return: Dictionary of results with detector names as keys and M_DM_max values as values.
		"""
		valid_WDs= []  # List of detector names with valid M_DM_max
		M_max = [] # List of valid M_DM_max values
		for name, mass, age in zip(self.det_names, self.det["mass"], self.det["age"]):
			try:
				max_DM_mass = self.M_DM_max(mass, age, alphat)
				if max_DM_mass:  # If a valid solution exists
					valid_WDs.append(name)  # Add detector name
					M_max.append(max_DM_mass[0])  # Add valid M_DM_max value
			except Exception as e:
				print(f"Error computing M_DM_max for {name}: {e}")
				#M_max.append(None)  # Append None for invalid values
		return valid_WDs, M_max

	def lam_trig_single(self, MWD):
		"""
		Compute trigger lengths of white dwarfs
		:return: trigger length of white dwarfs unit:cm
		"""
		rho = self.rho_WD(MWD)
		if rho <= self.rho1:
			return self.lam1 * (rho / self.rho1) ** (-2)
		elif self.rho1 < rho <= self.rho2:
			return self.lam1 * (rho / self.rho1) ** (np.log(self.lam2 / self.lam1) / np.log(self.rho2 / self.rho1))

	def lam_trig(self, rho):
		"""
		Compute trigger lengths of white dwarfs
		:return: trigger length of white dwarfs unit:cm
		"""
		if rho <= self.rho1:
			return self.lam1 * (rho / self.rho1) ** (-2)
		elif self.rho1 < rho <= self.rho2:
			return self.lam1 * (rho / self.rho1) ** (np.log(self.lam2 / self.lam1) / np.log(self.rho2 / self.rho1))


	def all_lam_trig(self, alphat):
		"""
		Compute trigger lengths of white dwarfs for which M_DM_max has valid solutions.
		: return: A tuple of two lists:
		- A list of detector names with valid trigger lengths.
		- A list of trigger lengths (in cm) corresponding to valid M_DM_max values.
		"""
		# Get valid detector names and M_DM_max values
		valid_detectors, valid_M_max = self.compute_all_M_DM_max(alphat)

		trigger_lengths = []  # List of trigger lengths for valid detectors

		for name in valid_detectors:
			try:
				# Find the mass of the corresponding detector
				index = list(self.det_names).index(name)  # Get index of the detector
				mass = self.det["mass"][index]  # Get mass
				rho = self.rho_WD(mass)  # Compute density, unit:g/cm^3

				# Compute trigger length based on density
				if rho <= self.rho1:
					length = self.lam1 * (rho / self.rho1) ** (-2)
				elif self.rho1 < rho <= self.rho2:
					length = self.lam1 * (rho / self.rho1) ** (
							np.log(self.lam2 / self.lam1) / np.log(self.rho2 / self.rho1))
				else:
					length = None  # If density is invalid, set to None

				trigger_lengths.append(length)  # Append trigger length
			except Exception as e:
				print(f"Error computing trigger length for {name}: {e}")
				trigger_lengths.append(None)  # Append None for invalid cases
		# Combine valid_M_max and trigger_lengths, and sort based on valid_M_max
		combined = sorted(zip(valid_M_max, trigger_lengths), key=lambda x: x[0])

		# Unzip the sorted pairs back into two separate lists
		sorted_M_max, sorted_trigger_lengths = zip(*combined)

		return list(sorted_M_max), list(sorted_trigger_lengths)

	def lambda_NS(self):

		MNS=1.58 #unit: sun mass
		rhoNS = 1e10 #density of carbon of Neutron star

		return self.lam_trig(rhoNS)

	def M_DM_NS(self, MNS, RNS, fx=0.01):
		"""
		MNS:mass of neutron star unit:sun mass
		RNS:radius of neutron star unit:km
		return: the maximum DM mass derived from neutron star constarint
		"""

		MNS_in_g = MNS*Msun

		r_dot = 8.5 # kpc
		r_s = 20.   # kpc
		rho_dot = 0.4 # GeV/cm^3
		rho_c = rho_dot*(r_dot/r_s)*(1+r_dot/r_s)**2 #GeV/cm^3

		def NFW_profile(r):

			return rho_c*(r_s/r)*(1/(1.+r/r_s))**2

		rNS = 1.2 #position of NS, kpc

		x = rNS / r_s
		mass = 4 * np.pi * rho_c * (r_s*kpc_to_cm)**3 * (np.log(1 + x) - x / (1 + x)) #GeV
		mass_in_g = mass*GeV_in_g

		G = 6.67430e-8           # gravitational constant (cm³/g/s²)

		v_DM_NS = np.sqrt(3*G*mass_in_g/(2*rNS*kpc_to_cm)) #DM velocity at NS, unit: cm s^-1

		v_esc_NS = np.sqrt(2*G*MNS_in_g/(2*RNS*1e5)) #unit: cm s^-1

		v_DM_NS_2 = v_DM_NS * 1e-5 / c #dimensionless
		v_esc_NS_2 = v_esc_NS * 1e-5 / c #dimensionless
		bcl_max = RNS*1e5/v_DM_NS_2 * np.sqrt(v_esc_NS_2 ** 2 / (1-v_esc_NS_2 ** 2))  #+ RDM cm
		#Gamma = rho_DM_NS / MDM_in_GeV * v_DM_NS * np.pi * bcl_max ** 2 * Gyr_in_s #Gyr^-1
		rho_DM_NS = NFW_profile(rNS) #GeV cm^-3
		tau_NS = 2.5*1e-9 #Gyr
		Gamma = 3/tau_NS /fx #Gyr^-1
		MDM_in_GeV = rho_DM_NS* v_DM_NS * np.pi * bcl_max ** 2 * Gyr_in_s/ Gamma
		MDM_in_g = MDM_in_GeV*GeV_in_g

		return MDM_in_g




def R_Fermball_mass(MF,mx,alpha_X):
	"""
	:param MF: g
	:param mx: GeV
	:param alpha_phi: 0.1
	:return: radius of the fermi ball, unit:cm
	"""

	MF_GeV = MF/GeV_in_g
	R = 3*alpha_X*MF_GeV/(2*mx**2)/cm_in_invGeV

	return R

def alpha_upper(MF, mx = 100):
	"""
	The maximum value of alpha_X derived from white dwarfs
	"""

	MF_GeV = MF / GeV_in_g

	return 2*mx**2*cm_in_invGeV/(3*MF_GeV)*np.sqrt(MF*(7.3*1e-11)/np.pi)

def alpha_upper_NS(MF, mx = 100):

	MF_GeV = MF / GeV_in_g

	return 2 * mx ** 2 * cm_in_invGeV / (3 * MF_GeV) * np.sqrt(MF * (2.5*1e-14) / np.pi)

def alpha_lower(MF, l_trig, mx = 100):

	MF_GeV = MF / GeV_in_g

	return l_trig*2*mx**2*cm_in_invGeV/(3*MF_GeV)

def self_interaction(MF, mx=100, lam=1e10, v=1e-2):
	"""
	The constraint from bullet cluster in the RX - MX figure
	"""

	righthand = (Msun/MF)**(1/2)*2*1e6*(v/1e-2)**2*np.exp(4e-3*(MF/Msun)**(1/2)*pc/lam)
	MF_GeV = MF / GeV_in_g
	num = (9/4)**2*righthand**2*3*np.pi*GCF**2*MF_GeV**3/mx**4

	return num**(1/5)/cm_in_invGeV


def self_interaction2(MF, mx=100, lam=1e10, v=1e-2):
	"""
	The constraint from bullet cluster in the alpha_X - MX figure
	"""
	righthand = (Msun/MF)**(1/2)*2*1e6*(v/1e-2)**2*np.exp(4e-3*(MF/Msun)**(1/2)*pc/lam)
	MF_GeV = MF / GeV_in_g
	num = righthand**2*4*np.pi*GCF**2*mx**6/MF_GeV**2/2

	return num**(1/5)

def CMB(MF, fx=0.01):
	"""
	The CMB constraint in the RX - MX figure
	"""
	return (3.3e-3/fx*MF/np.pi)**(1/2)


def lensing(MF):
	"""
	The microlensing constraint, MX = np.logspace(22.0, 28.0)
	"""
	return (1e-4/MF/np.pi)**(1/2)

def bmax_R(alpha, lam):
	"""
	Eq.(A6-A8) of 2209.03963
	"""
	GM_R = 0.2
	R = 10 #km
	v = 1e-3
	num = 2*GM_R*(1+alpha) + (GM_R*alpha)**2
	
	bmaxinner = R/v*np.sqrt(num/(1-GM_R))

	x = np.log(alpha/(lam*v**2/GM_R/R+1/np.log(alpha)))

	bmaxouter = np.sqrt(lam*x*(2*GM_R*R/v**2+lam*x))

	return np.minimum(bmaxinner, bmaxouter)/R

def alpha_low1(alpha, MX, lam=1e-1 * pc, fx= 0.01):
	"""
	Eq.(27) of 2209.03963
	"""
	return fx * 6e-22 * (Msun / MX) * bmax_R(alpha, lam)**2 * 1e-6 - 1

def alpha_low2(alpha, MX, lam=1e-1 * pc, fx=0.01):
	"""
	Eq.(31) of 2209.03963
	"""
	return fx*4e-4*(1+alpha)**2*(MX/Msun)* bmax_R(alpha, lam)**2 * 1e-6 - 1

def alpha_low_M1(M_values, lam=1e-1 * pc):
    
    result1 = []
    
    for M in M_values:
        a, b = 0, 1e30  
        solution = brentq(lambda alpha: alpha_low1(alpha, M, lam), a, b)
        delta_X = solution / 1e-6
        alpha_X = ((2 * np.pi * GCF**2 * 100**6 * delta_X**4) / (M / GeV_in_g)**2 )**(1/5)
        result1.append(alpha_X)
        #result1.append(solution)
    return np.array(result1)

def alpha_low_M2(M_values, lam=1e-1 * pc):
    
    result2 = []
    
    for M in M_values:
        a, b = 0, 1e15 
        solution = brentq(lambda alpha: alpha_low2(alpha, M, lam), a, b)
        delta_X = solution / 1e-6
        alpha_X = ((2 * np.pi * GCF**2 * 100**6 * delta_X**4) / (M / GeV_in_g)**2 )**(1/5)
        result2.append(alpha_X)
        #result2.append(solution)
    return np.array(result2)


def find_intersection(M1, result1, M2, result2):
    
    from scipy.interpolate import interp1d
    
    M_min = max(M1.min(), M2.min())
    M_max = min(M1.max(), M2.max())
    
    if M_min >= M_max:
        print("no overlap")
        return None, None
    
    M_common = np.logspace(np.log10(M_min), np.log10(M_max), 100)
    
    # 创建插值函数
    f1 = interp1d(M1, result1, kind='linear', bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(M2, result2, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    diff_func = lambda M: f1(M) - f2(M)
    
    M_test = np.logspace(np.log10(M_min), np.log10(M_max), 20)
    differences = diff_func(M_test)
    
    intersection_points = []
    
    for i in range(len(M_test) - 1):
        if differences[i] * differences[i + 1] < 0:
            try:
                intersection_M = brentq(diff_func, M_test[i], M_test[i + 1])
                intersection_alpha = f1(intersection_M)  
                intersection_points.append((intersection_M, intersection_alpha))
            except:
                continue
    
    if intersection_points:
       
        return intersection_points[0]
    else:
        print("no intersection point")
        return None, None




def main():
	pass



if __name__ == "__main__":
	obs = DW_analysis()
	print("begin")
	print(obs.Gamma_meet(1.4, 1e8, 10))
	print(obs.R_env(1.4,1))
	print(obs.env_condition(1.4,1))
	#print(obs.compute_all_env())
	print(max(obs.compute_all_env()))
	print("Neutron star:", obs.lambda_NS())
	print(obs.M_DM_NS(1.58,9.11))
	Mmax, trig = obs.all_lam_trig(alphat = 0)
	Mmax2, trig2 = obs.all_lam_trig(alphat = 100)
	Mmax_NS, trig_NS = obs.M_DM_NS(1.58,9.11), obs.lambda_NS()
	Mmin_NS = np.pi*trig_NS**2/(2.5*1e-14)
	Mmin_NS = float(Mmin_NS)
	Rmax_NS = np.sqrt(Mmax_NS * (2.5*1e-14) / np.pi)
	MNS = np.linspace(Mmin_NS, Mmax_NS, 100)

	print("Mmax:", max(Mmax))
	print(max(Mmax2))
	print(Mmax_NS)


	# Filter out None values from Mmax and trig
	filtered_data = [(m, t) for m, t in zip(Mmax, trig) if t is not None]
	if not filtered_data:
		print("No valid data points to plot.")
		exit()

	filtered_data2 = [(m, t) for m, t in zip(Mmax2, trig2) if t is not None]
	if not filtered_data2:
		print("No valid data points to plot.")
		exit()

	# Sort data by Mmax
	filtered_data = sorted(filtered_data, key=lambda x: x[0])
	Mmax, trig = zip(*filtered_data)
	print(min(trig))

	filtered_data2 = sorted(filtered_data2, key=lambda x: x[0])
	Mmax2, trig2 = zip(*filtered_data2)

# Find the lower boundary by iterating through the points
	lower_boundary = []
	for i, (m, t) in enumerate(zip(Mmax, trig)):
		# Add the first point and any point with a smaller trig value than previous
		while lower_boundary and lower_boundary[-1][1] > t:
			lower_boundary.pop()  # Remove points above the current lower boundary
		lower_boundary.append((m, t))

	# Unzip the lower boundary points into separate lists
	lower_Mmax, lower_trig = zip(*lower_boundary)



	print(min(lower_trig))

	lowest_trig = min(lower_trig)
	lowest_Mmax= min(lower_Mmax)
	highest_trig = max(lower_trig)
	highest_Mmax= max(lower_Mmax)
	Mmin = np.pi*lowest_trig**2/(7.3*1e-11)
	Mmin = float(Mmin)
	highest_Mmax = float(highest_Mmax)
	Mball = np.linspace(Mmin, highest_Mmax, 100)

	#same calculation for alphat = 10
	lower_boundary2 = []
	for i, (m, t) in enumerate(zip(Mmax2, trig2)):
		# Add the first point and any point with a smaller trig value than previous
		while lower_boundary2 and lower_boundary2[-1][1] > t:
			lower_boundary2.pop()  # Remove points above the current lower boundary
		lower_boundary2.append((m, t))

	# Unzip the lower boundary points into separate lists
	lower_Mmax2, lower_trig2 = zip(*lower_boundary2)




	lowest_trig2 = min(lower_trig2)
	lowest_Mmax2 = min(lower_Mmax2)
	highest_trig2 = max(lower_trig2)
	highest_Mmax2 = max(lower_Mmax2)
	Mmin2 = np.pi*lowest_trig2**2/(7.3*1e-11)
	Mmin2 = float(Mmin2)
	highest_Mmax2 = float(highest_Mmax2)
	Mball2 = np.linspace(Mmin2, highest_Mmax2, 100)

	Fermi_mass = np.logspace(-5, 25, 100)

	#lowest_trig = np.sqrt(M * (7.3*1e-11)/np.pi)
	# => M = (lowest_trig**2 * np.pi) / (7.3*1e-11

	intersect_M_horizontal = (lowest_trig**2 * np.pi) / (7.3 * 1e-11)
	intersect_M_horizontal2 = (lowest_trig2**2 * np.pi) / (7.3 * 1e-11)

	
	# R = np.sqrt(highest_Mmax * (7.3*1e-11)/np.pi)
	intersect_R_vertical = np.sqrt(highest_Mmax * (7.3 * 1e-11) / np.pi)
	intersect_R_vertical2 = np.sqrt(highest_Mmax2 * (7.3 * 1e-11) / np.pi)
  
	# Plot the original points and highlight the lower boundary
	plt.subplots_adjust(left=0.14,bottom=0.15)
	#plt.scatter(Mmax, trig, label="All Points", color="gray", alpha=0.6)
	plt.plot(lower_Mmax, lower_trig, color="#1f77b4", linestyle="-")
	plt.plot(Mball, np.sqrt(Mball*(7.3*1e-11)/np.pi), color="#1f77b4")
	plt.plot(lower_Mmax2, lower_trig2, color="#1f77b4", linestyle="dashed")
	plt.plot(Mball2, np.sqrt(Mball2*(7.3*1e-11)/np.pi), color="#1f77b4", linestyle="dashed")
	plt.plot(Fermi_mass,R_Fermball_mass(Fermi_mass,1e2,1e-15), color="purple",linestyle="solid",label=r"$\alpha_X = 10^{-15}$")
	plt.plot(Fermi_mass, R_Fermball_mass(Fermi_mass, 1e2, 1e-20), color="purple",linestyle="dashed", label=r"$\alpha_X = 10^{-20}$")
	plt.plot(Fermi_mass, R_Fermball_mass(Fermi_mass, 1e2, 1e-25), color="purple",linestyle="dotted", label=r"$\alpha_X = 10^{-25}$")
	plt.plot([intersect_M_horizontal, lowest_Mmax], [lowest_trig, lowest_trig], c="#1f77b4")
	plt.plot([highest_Mmax, highest_Mmax], [intersect_R_vertical, highest_trig], c="#1f77b4")
	plt.plot([intersect_M_horizontal2, lowest_Mmax2], [lowest_trig2, lowest_trig2], c="#1f77b4", linestyle="dashed")
	plt.plot([highest_Mmax2, highest_Mmax2], [intersect_R_vertical2, highest_trig2], c="#1f77b4", linestyle="dashed")
	plt.plot(MNS, np.sqrt(MNS*(2.5*1e-14)/np.pi), color="#ff7f0e")
	plt.plot([Mmin_NS, Mmax_NS], [trig_NS, trig_NS], c="#ff7f0e")
	plt.plot([Mmax_NS, Mmax_NS], [trig_NS, Rmax_NS], c="#ff7f0e")
	plt.text(7e4, 8e-3, r"$\sigma_{XA}/M_X = 7.3\times 10^{-11}~\mathrm{cm}^2/\mathrm{g}$", fontsize=12, rotation=30.5, color="#1f77b4")
	#plt.text(1e21, 8e-3, r"$N_{\mathrm{enc}}^{\mathrm{WD}}=3$", fontsize=12, rotation=90, color="#1f77b4")
	plt.text(1e19, 8e-3, r"$N_{\mathrm{enc}}^{\mathrm{WD}}=3, \tilde{\alpha} = 0$", fontsize=12, rotation=90, color="#1f77b4")
	plt.text(1e21, 8e-3, r"$N_{\mathrm{enc}}^{\mathrm{WD}}=3, \tilde{\alpha} = 100$", fontsize=12, rotation=90, color="#1f77b4")
	plt.plot(Fermi_mass,2*GCF*Fermi_mass/GeV_in_g/cm_in_invGeV,color='black')
	plt.text(3e20, 1e-8, r"$\mathrm{Black~Hole}$", fontsize=12, rotation=0, color="black")
	plt.text(4e20, 1e-9, r"$\mathrm{formation}$", fontsize=12, rotation=0, color="black")
	plt.plot(Fermi_mass, CMB(Fermi_mass), color = "gray")
	plt.text(5e7, 1e5, r"$\mathrm{CMB}$", fontsize=12, rotation=0, color="gray")
	#If f_X is larger than approximately 0.01, we have to take into account the constraint from bullet cluster
	#plt.plot(Fermi_mass, self_interaction(Fermi_mass,100),color="green",linestyle="dashed")
	#plt.text(1e9, 1e-4, r"$\mathrm{MICROSCOPE + bullet~cluster}$", fontsize=12, rotation=26, color="green")

	plt.title(r"$f_X = 0.01$", fontsize=18)
	
	fill_x = [intersect_M_horizontal]
	fill_y = [lowest_trig]

	for m, t in zip(lower_Mmax, lower_trig):
		fill_x.append(m)
		fill_y.append(t)

	fill_x.append(highest_Mmax)
	fill_y.append(intersect_R_vertical)
	

	plt.fill(fill_x, fill_y, color="#1f77b4", alpha=0.3, label="WD constraints")
	plt.fill([Mmin_NS,Mmax_NS,Mmax_NS], [trig_NS,trig_NS,Rmax_NS], color="#ff7f0e", alpha=0.3, label="NS constraints")
	# Add labels, grid, and legend
	plt.xlim([1e-3, 1e25])
	plt.ylim([1e-10, 1e7])
	plt.xlabel(r"$M_X~(\mathrm{g})$",fontsize=18)
	plt.ylabel(r"$R_X~(\mathrm{cm})$",fontsize=18)
	plt.tick_params(labelsize=14)
	plt.legend(fontsize=12, loc='upper left')
	plt.xscale('log')
	plt.yscale('log')
	

	plt.show()



