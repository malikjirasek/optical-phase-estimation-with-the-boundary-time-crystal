#------------------------------------------------------------------------------#
# This script computes the estimation error for an unknown optical phase shift #
# using the cascaded protocol described in the main text. The estimation error #
# is obtained from numerical diagonalization of the deformed Master            #
# equation. The script can be used to compute the data shwon in Fig. 4(a) and  #
# (b) by adjusting the parameters accordingly. The output is a csv  #file with #
# six columns: N,omega/omega_c, dphi, err (the estimation error), var (variance#
# of the intensity), abs_deriv (modulus of the derivative wrt. varphi).        #
# For Fig. 4(a): Ns=[11], ratios=[0.25,4], and                                 #
#                           dphis = 5*np.logspace(-3,-1,30,True,10)            #
# For Fig. 4(b): Ns=np.arange(5,21,2), ratios=[4], and                                 
#                           dphis = [0.005, 0.1]
# For Fig. 4(c): Ns=[6,11,16], ratios=[4], and
#                          dphis = np.logspace(-3,-1,10,True,10)                         
#------------------------------------------------------------------------------#
import numpy as np
from qutip import *
import pandas as pd
from scipy.sparse.linalg import eigs
from pathlib import Path
settings.core = CoreOptions(default_dtype='CSR')
#------------------------------------------------------------------------------#
ratios = [0.25,4] # omega/omega_c values to iterate over
dphis = np.logspace(-3,-1,30,True,10) # phase shift values to iterate over
Ns = [6] # system sizes to iterate over
# save in the same directory as this script (fallback to cwd if running interactively)
script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
filename = script_dir / f'est_err_abs_fig4a_N6.csv'
#------------------------------------------------------------------------------#
EST_ERR = np.array([np.zeros(6)]) #initialize array to store data, each row
                                  #corresponds to 
                                  #[N, omega/omega_c, dphi, err, var, abs_deriv]
#------------------------------------------------------------------------------#
for N in Ns: # iterate over system sizes (here only N=11)
    print('Calculating for N = ', N)
    # system parameters
    omega_c = N/2 # critical frequency for given system size N
    ds = 10**(-4) # small increment for s-parameter for numerical derivatives
    ddphi = 10**(-4) # small increment for phase shift for numerical derivatives
#------------------------------------------------------------------------------#
    # Collective spin operators 
    S_x = jmat(N/2, 'x')
    S_y = jmat(N/2, 'y')
    S_z = jmat(N/2, 'z')
    S_p = jmat(N/2, '+')
    S_m = jmat(N/2, '-')
    idenJ = qeye(S_z.shape[0])
    # operators of the sensor system S_1
    S_x1 = tensor(S_x, idenJ)
    S_x2 = tensor(idenJ, S_x)
    S_m1 = tensor(S_m, idenJ)
    # operators of the decoder system S_2
    S_m2 = tensor(idenJ, S_m)
    S_p1 = tensor(S_p, idenJ)
    S_p2 = tensor(idenJ, S_p)
    # identity
    idenJ = tensor(idenJ, idenJ)
#------------------------------------------------------------------------------#
    def L(ds, omega, dphi):
        """Function that builds the vectorized Lindblad Superoperator for the
        deformed Master Equation"""
        Ham = omega*(S_x1+S_x2)-0.5j*(np.exp(-1j*(dphi))*S_p2*S_m1 -
                                      np.exp(1j*(dphi))*S_p1*S_m2)
        HamT = (Ham.dag()).conj()
        Jump = S_m1+np.exp(1j*(dphi))*S_m2
        JumpD = Jump.dag()
        JpJm = JumpD*Jump
        JpJmT = (JpJm.dag()).conj()
        L = (-1j*(tensor(idenJ, Ham)-tensor(HamT, idenJ))
             +np.exp(-ds)*tensor(Jump.conj(), Jump)
             -0.5*(tensor(idenJ, JpJm))-0.5*(tensor(JpJmT, idenJ)))
        return L
#------------------------------------------------------------------------------#
    def dominant_eigval(L):
        """A function that diagonalizes a Master operator and returns its
        dominant eigenvalue"""
        L_sparse = L.data_as('csr_matrix')
        eval = eigs(L_sparse, k=1, which='LR', return_eigenvectors=False, ncv = 100)
        return np.real(eval[0])
#------------------------------------------------------------------------------#
    for ratio in ratios: # iterate over omega/omega_c values
        for dphi in dphis: # iterate over phase shift values
            omega = omega_c*ratio 
            #s+ds, dphi + ddphi
            LM = L(ds, omega, dphi+ddphi)
            lambdaPP = dominant_eigval(LM)
            #s-ds, phi-dphi
            LM = L(-ds, omega, dphi-ddphi)
            lambdaMM = dominant_eigval(LM)
            #s+ds, phi
            LM = L(ds, omega, dphi)
            lambdaP0 = dominant_eigval(LM)
            #s-ds, phi
            LM = L(-ds, omega, dphi)
            lambdaM0 = dominant_eigval(LM)
            var = abs((lambdaP0+lambdaM0)/(ds**2))
            abs_deriv = abs((lambdaPP-lambdaP0-lambdaM0+lambdaMM)/(2*ds*ddphi))
            err = np.sqrt(abs(var))/abs_deriv
            EST_ERR = np.append(EST_ERR,[[N,ratio,dphi,err,var,abs_deriv]], 
                            axis=0)
#------------------------------------------------------------------------------#
# Save data to csv file
EST_ERR = EST_ERR[1:,:] #remove initial zero row
df = pd.DataFrame(EST_ERR, columns=['N', 'ratio', 'phi', 'err', 'var', 
                                    'abs_deriv'])
df.to_csv(path_or_buf=str(filename))
print('Data saved to '+str(filename))