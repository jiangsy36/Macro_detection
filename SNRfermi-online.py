import numpy as np
from scipy import integrate
from scipy import interpolate
from time import time
import matplotlib.pyplot as plt
import os
from scipy.special import kn
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import pandas as pd
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

GCF = 6.70883e-39  # Gravitational constant in GeV^-2
mPL = GCF ** -0.5  # Planck mass in GeV

GeV_in_g = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm = 3.085677581e24  # 1 Mpc in cm
c = 299792.458  # in km/s
cm_in_invkeV = 5.067730938543699e7  # 1 cm in keV^-1
GeV_in_invs = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

Msun = 1.989e33  #g

det_names = ("LISA","TianQin","Taiji","BBO")
yr = 365.25*24*3600
det_tobs_yr = (4.5, 2.5, 5, 4)
det_tobs = (4.5*yr, 2.5*yr, 5.*yr, 4.*yr)
det_Len = (2.5*1e6, np.sqrt(3) * 1e5, 3 * 1e6, 5*1e4) #km
det_fstar = (c/(2*np.pi*det_Len[0]), c/(2*np.pi*det_Len[1]), c/(2*np.pi*det_Len[2]), c/(2*np.pi*det_Len[3]))
lower_lim = (1e-4, 1e-4, 1e-4, 1e-1)
upper_lim = (1, 1, 1, 10)
Poms_s = (15*1e-15, 1*1e-15, 8*1e-15) #km
Pacc_s = (3*1e-18, 1*1e-18, 3*1e-18) #km s^-2

target_SNR = 10

delta_SM = 1e-6

V = 220 #km/s, DM velocity

rho_DM = 0.4 #GeV cm^-3

fX = 0.01 #DM fraction

mX = 100 #mass of DM particle

color_dict = {
    "LISA": 'blue',
    "TianQin": 'red', 
    "Taiji": 'green',
    "BBO": 'purple'
}


def noise_PSD(f, name):
    """
    we choose the X channel
    """
    detector_index = det_names.index(name)
    #f_star = det_fstar[detector_index]
    L = det_Len[detector_index] #km
    if name == "BBO":
        Soms = 2*1e-34/(3*L**2*1e6)
        Sacc = 4.5*1e-34/((2*np.pi*f)**4*(3*L*1e3)**2)
        Sn = 16*np.sin(2*np.pi*f*L/c)*np.sin(2*np.pi*f*L/c)*((3+np.cos(4*np.pi*f*L/c))*Sacc + Soms)
    
    elif name in ["LISA", "TianQin", "Taiji"]:
        Poms = Poms_s[detector_index]
        Pacc = Pacc_s[detector_index]
        Soms = (2*np.pi*f*Poms/c)**2*(1+(2e-3/f)**4)
        Sacc = (Pacc/2/np.pi/f/c)**2*(1+(4e-4/f)**2)*(1+(f/8e-3)**4)
        Sn = 16*np.sin(2*np.pi*f*L/c)*np.sin(2*np.pi*f*L/c)*((3+np.cos(4*np.pi*f*L/c))*Sacc + Soms)

    return Sn


def P_f(f, M, V, D, alpha, name):
    """
    D: km
    V:km/s
    """
    detector_index = det_names.index(name)
    f_star = det_fstar[detector_index]
    L = det_Len[detector_index] #km
    M_in_GeV = M / GeV_in_g
    V_ns = V/c
    D_in_invs = D/c
    def Dprime(D):
        if D<L:
            return (D**2+L**2)/2/L 
        elif D>=L:
            return D
    transfer = (f/f_star)**2/(1+(f/f_star)**2)
    P = 32/(3*np.pi)*(GCF*(1+alpha)*M_in_GeV/V_ns**2)**2*np.sin(f/f_star)**2*((kn(0,D*2*np.pi*f/V)*np.cos(f/f_star)-kn(0,(D+L)*2*np.pi*f/V))**2 + (kn(1,D*2*np.pi*f/V)*np.cos(f/f_star)-kn(1,(D+L)*2*np.pi*f/V))**2)
    #Palpha = 8/(3*np.pi)*(GCF*(1+alpha)*M_in_GeV/V_ns**2)**2*np.sin(f/f_star/2)**2*((kn(0,D*2*np.pi*f/V)*(1+2*np.cos(f/f_star))-kn(0,(D+L)*2*np.pi*f/V))**2 + (kn(1,D*2*np.pi*f/V)*(1+2*np.cos(f/f_star))-kn(1,(D+L)*2*np.pi*f/V))**2)
    #Pzeta = 8/(3*np.pi)*(GCF*(1+alpha)*M_in_GeV/V_ns**2)**2*np.sin(f/f_star/2)**2*((kn(0,D*2*np.pi*f/V)-kn(0,(D+L)*2*np.pi*f/V))**2 + (kn(1,D*2*np.pi*f/V)-kn(1,(D+L)*2*np.pi*f/V))**2)

    
    return P/GeV_in_invs**2


def plot_noise(M, V, D, alpha):

    for name in ["LISA", "TianQin", "Taiji", "BBO"]:
        detector_index = det_names.index(name)
        lowerlim = lower_lim[detector_index]
        upperlim = upper_lim[detector_index]
        color = colors[detector_index]

        plt.subplots_adjust(left=0.14,bottom=0.15)
        f = np.linspace(lowerlim,upperlim, 1000)
        f2 = np.linspace(1e-4, 0.1, 1000)
        plt.plot(f,noise_PSD(f,name), c= color,label=name)
        #plt.plot(f,noise_PSD_2(f,name), c= color,label=name)
        plt.plot(f2, 4*2*np.pi*f2*P_f(f2, M, 220, D, alpha, name), c= color, ls='dashed')
    
    
    plt.plot(f2, 4*2*np.pi*f2*P_f(f2,1e0, 220, 50000, 10, "LISA"), c= 'black',ls='dashed',label=r"$M_X = 10^{15}~\mathrm{g}, \tilde{\alpha} = 10$")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-52,1e-26)
    plt.xlim(1e-4,10)
    plt.xlabel(r"$f~(\mathrm{Hz})$", fontsize=18)
    plt.ylabel(r"$8\pi f P(f)~(\mathrm{Hz}^{-1})$", fontsize=18)
    plt.legend(loc='upper left',fontsize=12)
    plt.title(r"$D = 50000~\mathrm{km}$", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.show()


def int_SNR(f, M, V, D, alpha, name):
    sensitivity = noise_PSD(f,name)
    Pf = P_f(f, M, V, D, alpha, name)

    return Pf / sensitivity


def calc_SNR(M, V, D, alpha, name):
    """
    SNR as functions of M and tilde_alpha
    """

    detector_index = det_names.index(name)
    lowerlim = lower_lim[detector_index]
    upperlim = upper_lim[detector_index]
    f = np.logspace(np.log10(lowerlim),np.log10(upperlim),1000)
    int = int_SNR(f, M, V, D, alpha, name)
    frac = np.array(int)
    integral = integrate.simpson(frac, f)
    tobs = det_tobs[detector_index]
    snr2 = 2. * integral

    return np.sqrt(snr2)


def calc_SNR_alphaX(M, V, D, alphaX, name):
    """
    SNR as functions of M and alphaX
    """

    detector_index = det_names.index(name)
    lowerlim = lower_lim[detector_index]
    upperlim = upper_lim[detector_index]
    f = np.logspace(np.log10(lowerlim),np.log10(upperlim),1000)
    delta_X = (2*alphaX**5*(M/GeV_in_g)**2/(4*np.pi*GCF**2*mX**6))**(1/4)
    alpha = delta_SM*delta_X
    int = int_SNR(f, M, V, D, alpha, name)
    frac = np.array(int)
    integral = integrate.simpson(frac, f)
    tobs = det_tobs[detector_index]
    snr2 = 2. * integral

    return np.sqrt(snr2)


def bisect_alphaX(M, V, D, detector, target_SNR=10, alphaX_min=1e-20, alphaX_max=1e-7, 
                 tol=1e-3, max_iter=50, debug=False):
    
    def f(alphaX):
        try:
            current_SNR = calc_SNR_alphaX(M, V, D, alphaX, detector)
            return current_SNR - target_SNR
        except:
            return -target_SNR  
    
    f_min = f(alphaX_min)
    f_max = f(alphaX_max)
    
    if debug:
        print(f"  alphaX range: [{alphaX_min:.2e}, {alphaX_max:.2e}]")
        print(f"  f(alphaX_min) = {f_min:.6f}, f(alphaX_max) = {f_max:.6f}")
    
    if f_min * f_max > 0:
        if debug:
            print(f"  no solution in range: f(min)={f_min:.6f}, f(max)={f_max:.6f}")
        return None
    
    low, high = alphaX_min, alphaX_max
    for i in range(max_iter):
        mid = (low + high) / 2
        f_mid = f(mid)
        
        if debug and i < 5:  
            print(f"  iteration {i}: alphaX = {mid:.2e}, f = {f_mid:.6f}")
        
        if abs(f_mid) < tol:
            if debug:
                print(f"  Convergence: alphaX = {mid:.2e}, SNR = {f_mid + target_SNR:.6f}")
            return mid
        
        if f_mid * f(low) < 0:
            high = mid
        else:
            low = mid
    
    final_alphaX = (low + high) / 2
    final_SNR = calc_SNR_alphaX(M, V, D, final_alphaX, detector)
    
    if abs(final_SNR - target_SNR) < tol:
        if debug:
            print(f"  Reached the maximum number of iterations but met the accuracy: alphaX = {final_alphaX:.2e}, SNR = {final_SNR:.6f}")
        return final_alphaX
    else:
        if debug:
            print(f"  No convergence: alphaX = {final_alphaX:.2e}, SNR = {final_SNR:.6f} (target SNR: {target_SNR})")
        return None


M_values_log = np.linspace(1, 22, 250) 
M_values = 10 ** M_values_log

detectors = ["LISA", "TianQin", "Taiji"]
data_records = []

print("Begin parameter scanning...")
print("=" * 60)

for i, M in enumerate(M_values):
    print(f"M = {M:.2e} g ({i+1}/{len(M_values)})")
    
    for detector in detectors:
        detector_index = det_names.index(detector)
        tobs = det_tobs_yr[detector_index]
        eta_obs = 1/tobs  # yr^-1

        pre = np.pi * rho_DM * V * (1e5)**3 * yr * GeV_in_g / M * fX
        D = np.sqrt(eta_obs / pre)
        
        debug_mode = (i % 10 == 0)  
        alphaX_solution = bisect_alphaX(M, V, D, detector, target_SNR, 
                                      alphaX_min=1e-20, alphaX_max=1e-7,
                                      tol=1e-4, debug=debug_mode)
        
        if alphaX_solution is not None:
            
            final_SNR = calc_SNR_alphaX(M, V, D, alphaX_solution, detector)
            
            data_records.append({
                'M_g': M,
                'alpha_X': alphaX_solution,
                'detector': detector,
                'D_km': D,
                'eta': eta_obs,
                'SNR': final_SNR
            })
            
            print(f"  {detector}: alphaX = {alphaX_solution:.2e}, D = {D:.2f} km, SNR = {final_SNR:.2f}")
        else:
            print(f"  {detector}: has no solution within the range")
    
    print("-" * 40)

print("Scan complete!")
print(f"Total Data Points: {len(data_records)}")

if data_records:
    df = pd.DataFrame(data_records)
    output_filename = "M_alphaX_scan_bisect_results_001.xlsx"
    df.to_excel(output_filename, index=False, float_format="%.6e")
    print(f"Data has been exported to: {output_filename}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.14, bottom=0.15)
    
    for detector in detectors:
        detector_data = df[df['detector'] == detector]
        if len(detector_data) > 0:
            ax.loglog(detector_data['M_g'], detector_data['alpha_X'], 
                      label=detector, color=color_dict[detector])
    
    ax.set_xlabel(r"$M_X~(\mathrm{g})$", fontsize=18)
    ax.set_ylabel(r"$\alpha_X$", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(r"$\mathrm{SNR} = 10$", fontsize=18)
    plt.show()
else:
    print("No valid data points found")



