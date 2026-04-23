#------------------------------------------------------------------------------#
# This script computes the classical Fisher information (FI) of the           #
# photon-counting record for the perfect absorber (cascaded source-decoder)    #
# protocol. The computation follows the trajectory-based Monte Carlo method    #
# described in Sec. VII of the Supplemental Material (Eqs. (S76)-(S82)),       #
# adapted to photon counting: each trajectory is generated using the Kraus     #
# operators K_0 (no click) and K_1 (click) of the cascaded master equation,    #
# and the auxiliary vector |phi_t> is propagated alongside |psi_t>. The FI     #
# rate is estimated from the time-averaged slope of the cumulative FI along    #
# each trajectory, and the ensemble average is taken over many realizations.   #
# The output is a csv file with columns: N, ratio (omega/omega_c), dphi,       #
# av_FI_rate, std_FI_rate, and n_trajectories. The data complements the       #
# estimation error shown in Fig. 4 and can be plotted alongside it.            #
#------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
from qutip import jmat, qeye, tensor, settings, CoreOptions
from scipy.sparse import eye as sp_eye
from pathlib import Path
import concurrent.futures
settings.core = CoreOptions(default_dtype='CSR')
#------------------------------------------------------------------------------#
# Parameters to iterate over. Adjust these to explore the phase diagram.       #
# By default, we match the parameter region of Fig. 4 (omega/omega_c = 4,      #
# small Delta phi in the time-crystal regime, and omega/omega_c = 0.25 in the  #
# stationary regime).                                                          #
#------------------------------------------------------------------------------#
Ns        = [11]#np.arange(5,21,2)             # system sizes to iterate over
ratios    = [0.25, 4.0]                   # omega/omega_c values
dphis     = np.array([0.01, 0.05, 0.5]) # phase differences Delta phi = phi-phi'
n_traj    = 150000                           # number of trajectories
kappa     = 1.0                           # emission rate (units)
dt        = 0.01 / kappa                  # discrete time step
T_tot     = 40.0 / kappa                  # total trajectory duration
T_warmup  = 15.0 / kappa                  # discard cumulative FI before T_warmup
rng_seed  = 1234
#------------------------------------------------------------------------------#
script_dir = Path(__file__).resolve().parent if '__file__' in globals() \
             else Path.cwd()
filename   = script_dir / 'FI_count_cascaded.csv'
#------------------------------------------------------------------------------#
def build_operators(N):
    """Return the collective operators of source (S_*1) and decoder (S_*2)
    acting on the joint Hilbert space of dimension (N+1)^2, as sparse
    matrices (scipy.sparse.csr_matrix)."""
    S_x = jmat(N/2, 'x')
    S_p = jmat(N/2, '+')
    S_m = jmat(N/2, '-')
    idenJ = qeye(S_x.shape[0])
    S_x1 = tensor(S_x,   idenJ).data_as('csr_matrix')
    S_x2 = tensor(idenJ, S_x  ).data_as('csr_matrix')
    S_p1 = tensor(S_p,   idenJ).data_as('csr_matrix')
    S_p2 = tensor(idenJ, S_p  ).data_as('csr_matrix')
    S_m1 = tensor(S_m,   idenJ).data_as('csr_matrix')
    S_m2 = tensor(idenJ, S_m  ).data_as('csr_matrix')
    Id   = tensor(idenJ, idenJ).data_as('csr_matrix')
    return S_x1, S_x2, S_p1, S_p2, S_m1, S_m2, Id

df = pd.DataFrame(records,
                  columns=['N', 'ratio', 'phi', 'av_FI_rate',
                           'mc_err_FI_rate', 'n_trajectories'])
df.to_csv(path_or_buf=str(filename), index=False)
print(f'Data saved to {filename}')
