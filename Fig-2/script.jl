using LinearAlgebra
using SparseArrays 
using Dates
using Plots
using DelimitedFiles
using Statistics

include("trajectories.jl")
include("models.jl")

############################################ Main #############################

name_file="./FI_t_params_7_dt7_N30.dat"
name_fig="./fig_params_7_dt7.pdf"
NS = 30
dOmega = 0.0001
dvarphi = 0.001
t_final = 40
timespan = 0:1:t_final
kappa = 1
Omega_c = kappa*NS/2
Omega =2.6*Omega_c 
varphi = 1.570796327
phase_homodyne = 1.570796327
dt = 0.000025 # Change for smaller/larger stepsize
n_save = 4000
Ntime = Int(floor(t_final/dt)) # Number of timesteps
timespan_fisher = (1 : Ntime) * dt
num_traj = 1000
##### Initial state:
theta = 0
phi = 0
init_state_spins = Dicke_state(theta, phi, NS/2)
init_state_spins = init_state_spins/norm(init_state_spins)
init_state = init_state_spins


println("dt: ",dt)
println("w over wc: ",Omega/Omega_c)


# Initialize files
#label = "data"*string(task_id)*".h5"   
#data = joinpath( folderpath, label )

time0=time()


# Construct relevant objects
H, Hdomega, Hdvarphi, L, dLomega, dLvarphi, expectation_values, identity = BTC_model(NS, Omega, kappa, varphi)
psi0 = init_state
ops = expectation_values
# Determine FI for ideal homodyne monitoring
t_full, exp_val1, exp_val2, current, FisherT, ErrorFisher = Fisher_trajectory_homodyne_jump_optimized( H, L, dLvarphi, num_traj, psi0, ops, t_final, dt, phase_homodyne, n_save )


time1=time()
println("Run time: ",time1-time0)

i = Int(floor(15/(n_save*dt)))
println("Mean: $(mean(FisherT[i:end]./ t_full[i:end]))")
println("Std:  $(mean(ErrorFisher[i:end]./ t_full[i:end]))")
println("QFI Stationary:", 4*Omega*Omega/kappa)
println("QFI TC asymptotic:", kappa*NS*(NS+2)*((NS-1)*(NS+3)/135+2/3))


writedlm(name_file,hcat(t_full,FisherT,ErrorFisher))


