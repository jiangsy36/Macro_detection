import numpy as np
from scipy.special import kn
from scipy.linalg import svd
import pandas as pd

GCF = 6.70883e-39
GeV_in_g = 1.782661907e-24
c = 299792.458
cm_in_invkeV = 5.067730938543699e7
GeV_in_invs = cm_in_invkeV * c * 1.e11

def calculate_correlation_matrix(cov_matrix, param_names):

    try:
        
        std_devs = np.sqrt(np.diag(cov_matrix))
        
      
        outer_std_devs = np.outer(std_devs, std_devs)
        

        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = cov_matrix / outer_std_devs
        
        corr_matrix[np.isnan(corr_matrix)] = 0
        
        
        np.fill_diagonal(corr_matrix, 1.0)
        
  
        corr_df = pd.DataFrame(corr_matrix, index=param_names, columns=param_names)
        
        return corr_df
    except Exception as e:
 
        return None

class SpaceBasedFisherCalculator:
    def __init__(self, detector_type='LISA', f_min=1e-4, f_max=1.0, n_freq=50000):
        self.detector_type = detector_type
        self.f_min = f_min
        self.f_max = f_max
        
   
        self.set_detector_parameters()
  
        self.frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
        
        self.eps = 1e-5
        self.eps_mass = 1e-8
        
      
        self.components = [self]
    
    def set_detector_parameters(self):
     
        det_names = ("LISA", "TianQin", "Taiji", "BBO")
        detector_index = det_names.index(self.detector_type)
        
        det_Len = (2.5e6, np.sqrt(3) * 1e5, 3e6, 5e4)
        self.L = det_Len[detector_index]
        
        det_fstar = (c / (2 * np.pi * det_Len[0]), 
                     c / (2 * np.pi * det_Len[1]), 
                     c / (2 * np.pi * det_Len[2]), 
                     c / (2 * np.pi * det_Len[3]))
        self.f_star = det_fstar[detector_index]
        
        lower_lim = (1e-4, 1e-4, 1e-4, 1e-1)
        upper_lim = (1, 1, 1, 10)
        self.f_lower = lower_lim[detector_index]
        self.f_upper = upper_lim[detector_index]
        
        Poms_s = (15e-15, 1e-15, 8e-15)
        Pacc_s = (3e-18, 1e-18, 3e-18)
        
        if self.detector_type == "BBO":
            self.Poms = None
            self.Pacc = None
        else:
            lisa_tq_tj_index = det_names.index(self.detector_type)
            self.Poms = Poms_s[lisa_tq_tj_index]
            self.Pacc = Pacc_s[lisa_tq_tj_index]
        
        self.c = c
    
    def Sn_channels(self, f):

        tiny = np.finfo(f.dtype).tiny
        
        if self.detector_type == "BBO":
           
            Soms = 2e-34 / (3 * self.L**2 * 1e6)
            Sacc = 4.5e-34 / ((2 * np.pi * f)**4 * (3 * self.L * 1e3)**2)
            
            arg_pi_f_L_c = np.pi * f * self.L / self.c
            
       
            sin_2pifL = np.sin(2 * arg_pi_f_L_c)
            cos_4pifL = np.cos(4 * arg_pi_f_L_c)
            Sn_X = 16 * (sin_2pifL**2) * ((3 + cos_4pifL) * Sacc + Soms)
            
           
            sin_1pifL = np.sin(arg_pi_f_L_c)
            sin_3pifL = np.sin(3 * arg_pi_f_L_c)
            Sn_alpha = 8 * (2 * (sin_1pifL**2) + (sin_3pifL**2)) * Sacc + 6 * Soms
            
        
            Sn_zeta = 6 * (4 * (sin_1pifL**2) * Sacc + Soms)
        else:
         
            Soms = (2 * np.pi * f * self.Poms / self.c)**2 * (1 + (2e-3 / f)**4)
            Sacc = (self.Pacc / 2 / np.pi / f / self.c)**2 * (1 + (4e-4 / f)**2) * (1 + (f / 8e-3)**4)
            
            arg1 = np.pi * f * self.L / self.c
            arg2 = 2 * arg1
            arg3 = 3 * arg1
            
            transfer_1 = (arg1**2) / (1 + arg1**2)
            transfer_2 = (arg2**2) / (1 + arg2**2)
            transfer_3 = (arg3**2) / (1 + arg3**2)
            
    
            Sn_X = 16 * transfer_2 * ((3 + np.cos(4 * np.pi * f * self.L / self.c)) * Sacc + Soms)
            
           
            Sn_alpha = 8 * (2 * transfer_1 + transfer_3) * Sacc + 6 * Soms
            
          
            Sn_zeta = 6 * (4 * transfer_1 * Sacc + Soms)
        
        return Sn_X + tiny, Sn_alpha + tiny, Sn_zeta + tiny
    
    def Sn(self, f):
    
        Sn_X, Sn_alpha, Sn_zeta = self.Sn_channels(f)
        Sn_total = 1.0 / (1.0/Sn_X + 1.0/Sn_alpha + 1.0/Sn_zeta)
        return Sn_total
    
    def signal_power_spectrum_channels(self, params, f):

        log10_M, log10_D, log10_alpha, V = params
        M_in_GeV = 10**log10_M / GeV_in_g
        D = 10**log10_D
        alpha_X = 10**log10_alpha
        delta_X = (2*alpha_X**5*(M_in_GeV)**2/(4*np.pi*GCF**2*100**6))**(1/4)
        delta_SM = 1e-6
        alpha = delta_SM*delta_X
        V_ns = V / self.c
        
    
        common_factor = (GCF * (1 + alpha) * M_in_GeV / V_ns**2)**2
        
  
        kn0_D = kn(0, D * 2 * np.pi * f / V)
        kn0_DL = kn(0, (D + self.L) * 2 * np.pi * f / V)
        kn1_D = kn(1, D * 2 * np.pi * f / V)
        kn1_DL = kn(1, (D + self.L) * 2 * np.pi * f / V)
     
        P_X = (32 / (3 * np.pi) * common_factor * np.sin(f / self.f_star)**2 * 
               ((kn0_D * np.cos(f / self.f_star) - kn0_DL)**2 + 
                (kn1_D * np.cos(f / self.f_star) - kn1_DL)**2))
        
        P_alpha = (8 / (3 * np.pi) * common_factor * np.sin(f / self.f_star / 2)**2 * 
                   ((kn0_D * (1 + 2*np.cos(f / self.f_star)) - kn0_DL)**2 + 
                    (kn1_D * (1 + 2*np.cos(f / self.f_star)) - kn1_DL)**2))
      
        P_zeta = (8 / (3 * np.pi) * common_factor * np.sin(f / self.f_star / 2)**2 * 
                  ((kn0_D - kn0_DL)**2 + (kn1_D - kn1_DL)**2))
        
   
        P_X = P_X / GeV_in_invs**2
        P_alpha = P_alpha / GeV_in_invs**2
        P_zeta = P_zeta / GeV_in_invs**2
        
        return P_X, P_alpha, P_zeta
    
    def signal_amplitude_channels(self, params, f):

        P_X, P_alpha, P_zeta = self.signal_power_spectrum_channels(params, f)
        return np.sqrt(P_X), np.sqrt(P_alpha), np.sqrt(P_zeta)
    
    def signal_amplitude(self, params, f):

        h_X, _, _ = self.signal_amplitude_channels(params, f)
        return h_X
    
    def _get_derivative_channels(self, params, param_index):

        params_plus = np.array(params, dtype=float)
        params_minus = np.array(params, dtype=float)
        
        x_i = params[param_index]
        
       
        if param_index in [0, 1, 2]:
            step = self.eps_mass * x_i if x_i != 0 else self.eps_mass
        else:
            step = self.eps * x_i if x_i != 0 else self.eps
        
        params_plus[param_index] += step
        params_minus[param_index] -= step
        
       
        h_X_plus, h_alpha_plus, h_zeta_plus = self.signal_amplitude_channels(params_plus, self.frequencies)
        h_X_minus, h_alpha_minus, h_zeta_minus = self.signal_amplitude_channels(params_minus, self.frequencies)
        
      
        deriv_X = (h_X_plus - h_X_minus) / (2 * step)
        deriv_alpha = (h_alpha_plus - h_alpha_minus) / (2 * step)
        deriv_zeta = (h_zeta_plus - h_zeta_minus) / (2 * step)
        
        return deriv_X, deriv_alpha, deriv_zeta
    
    def _inner_product_three_channels(self, a_channels, b_channels):
       
        a_X, a_alpha, a_zeta = a_channels
        b_X, b_alpha, b_zeta = b_channels
        
       
        Sn_X, Sn_alpha, Sn_zeta = self.Sn_channels(self.frequencies)
        
        integrand_X = (a_X * b_X) / Sn_X
        integrand_alpha = (a_alpha * b_alpha) / Sn_alpha
        integrand_zeta = (a_zeta * b_zeta) / Sn_zeta
        
     
        integral_X = np.trapz(integrand_X, self.frequencies)
        integral_alpha = np.trapz(integrand_alpha, self.frequencies)
        integral_zeta = np.trapz(integrand_zeta, self.frequencies)
        
       
        total_inner_product = 4 * (integral_X + integral_alpha + integral_zeta)
        
        return total_inner_product
    
    def calculate_fisher_matrix(self, params):
        
        n_params = len(params)
        fisher_matrix = np.zeros((n_params, n_params))
        
    
        derivatives_all_channels = []
        for i in range(n_params):
            deriv_X, deriv_alpha, deriv_zeta = self._get_derivative_channels(params, i)
            derivatives_all_channels.append((deriv_X, deriv_alpha, deriv_zeta))
        
    
        for i in range(n_params):
            for j in range(i, n_params):
                
                inner_prod = self._inner_product_three_channels(
                    derivatives_all_channels[i],
                    derivatives_all_channels[j]
                )
                fisher_matrix[i, j] = inner_prod
                if i != j:
                    fisher_matrix[j, i] = inner_prod
        
        return fisher_matrix
    
    def calculate_covariance_matrix(self, fisher_matrix, svd_cutoff=1e-15):
     
        try:
            U, s, Vh = svd(fisher_matrix)
            s_inv = np.zeros_like(s)
            valid_s = s > svd_cutoff
            s_inv[valid_s] = 1.0 / s[valid_s]
            covariance_matrix = Vh.T @ np.diag(s_inv) @ U.T
            return covariance_matrix
        except np.linalg.LinAlgError:
          
            n_params = fisher_matrix.shape[0]
            return np.full((n_params, n_params), np.inf)
    
    def get_parameter_errors(self, covariance_matrix):
       
        diag = np.diag(covariance_matrix).copy()
        diag[diag < 0] = np.inf
        errors = np.sqrt(diag)
        return errors