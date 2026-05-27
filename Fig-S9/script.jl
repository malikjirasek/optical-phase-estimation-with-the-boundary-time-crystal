using Distributed

if nprocs() == 1
    addprocs(Sys.CPU_THREADS - 1)
end

@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
@everywhere using SparseArrays
@everywhere using FastExpm
@everywhere using ProgressMeter
using Dates
using Plots
using DelimitedFiles
using Statistics

@everywhere include("trajectories.jl")
@everywhere include("models.jl")


############################################ Main #############################

#name_file="./N30/FI_t_params_7_dt7_N30.dat"
name_fig="./N6/fig_params_7_dt7.pdf"
NS = 6 # Change for other parameters [10,15]
name_file="./FI_time_resolved_N$(NS)_ratio$(4).dat"
dOmega = 0.0001
dvarphi = 0.001
t_final = 2e4
timespan = 0:1:t_final
kappa = 1
Omega_c = kappa*NS/2
Omega1 = 4*Omega_c # Change to 0.1 for stationary regime 
Omega2 = 1.0*Omega1
varphi = 1.570796327
deltavarphi=0.005
phase_homodyne = 1.570796327
dt = 0.01 # Change for smaller/larger stepsize
n_save = 50
Ntime = Int(floor(t_final/dt)) # Number of timesteps
timespan_fisher = (1 : Ntime) * dt
num_traj = 1000 # Number of trajectories to average over for the Fisher information. Change for more/less trajectories.
# Change to (theta,phi)=(pi/2,0) etc. for FIGSM2 (a-c)
theta = 0
phi = 0
init_state_spins = Dicke_state(theta, phi, NS/2)
init_state_spins = init_state_spins/norm(init_state_spins)
init_state = kron(init_state_spins,init_state_spins)


println("dt: ",dt)
println("w over wc: ",Omega1/Omega_c)


# Initialize files
#label = "data"*string(task_id)*".h5"   
#data = joinpath( folderpath, label )

time0=time()


# Construct relevant objects
H, Hdomega, Hdvarphi, L, dLomega, dLvarphi, expectation_values, _ = BTC_cascaded_model(NS, Omega1, Omega2, kappa, deltavarphi)
psi0_SB = init_state
ops_SB = expectation_values
# Determine FI and QFI for ideal photocounting
println("Starting Fisher_trajectory_reduced with num_traj = ", num_traj)
t_SB, exp_val1_SB, exp_val2_SB, emissions_SB, FisherT_SB_omega, ErrorFisher = Fisher_trajectory_reduced( H, Hdvarphi, L, dLvarphi, num_traj, psi0_SB, ops_SB, t_final, dt,n_save)


time1=time()
println("Run time: ",time1-time0)

i = Int(floor(2000/(n_save*dt)))
println("Mean: $(mean(FisherT_SB_omega[i:end]./ t_SB[i:end]))")
println("Std:  $(mean(ErrorFisher[i:end]./ t_SB[i:end]))")
println("QFI Stationary:", 4*Omega1*Omega1/kappa)
println("QFI TC asymptotic:", kappa*NS*(NS+2)*((NS-1)*(NS+3)/135+2/3))


writedlm(name_file,hcat(t_SB,FisherT_SB_omega,ErrorFisher))
println("Saved time-resolved data to ", name_file)

exit() # Exit before sweeping so we don't accidentally do a sweep if not intended.

#write(file, "Fisher_traj",  FisherT_SB_omega)
#write(file, "QFisher_traj",  QFisherT_SB_omega)
#write(file, "real_traj",  real_part)
#write(file, "imag_part",  imag_part)

#p1 = plot(t_SB,exp_val1_SB)
#plot!(p1,t_SB,exp_val2_SB)

#p2 = plot(t_SB, FisherT_SB_omega)

#p2 =plot(t_SB[2:end], FisherT_SB_omega[2:end] ./ t_SB[2:end], ribbon = ErrorFisher[2:end]./ t_SB[2:end],fillalpha = 0.25,color = :blue)
#hline!(p2, [4*Omega*Omega/kappa])
#hline!(p2, [4*Omega*Omega*cos(varphi-phase_homodyne)*cos(varphi-phase_homodyne)/kappa])
#hline!(p2, [4])

#p3 = plot(t_SB, emissions_SB)

#display(plot(p1,p2,p3, layout= (3,1)))
#savefig(name_fig)
#println("Press Enter to exit...")
#readline()

############################################ deltavarphi sweep ################
#deltavarphi_values = exp10.(range(log10(0.001), log10(0.1), length=2))
#t_stat     = 2000.0
#t_stat_idx = Int(floor(t_stat / (n_save * dt)))
#
#csv_path = "./FI_deltavarphi_sweep_N$(NS)_ratio$(Omega1/Omega_c)_singletraj.csv"
#open(csv_path, "w") do io
#    println(io, ",N,ratio,deltavarphi,mean_FI_rate,std_FI_rate")
#end
#println("Checkpoint file: $csv_path")
#
#for (k, dφ) in enumerate(deltavarphi_values)
#    println("[$k/$(length(deltavarphi_values))] deltavarphi = $dφ")
#    time_sw = time()
#
#    H_sw, _, Hdvarphi_sw, L_sw, _, dLvarphi_sw, ops_sw, _ =
#        BTC_cascaded_model(NS, Omega1, Omega2, kappa, dφ)
#
#    t_sw, _, _, _, FI_sw, err_sw =
#        Fisher_trajectory_reduced(H_sw, Hdvarphi_sw, L_sw, dLvarphi_sw,
#                                   num_traj, psi0_SB, ops_sw, t_final, dt, n_save)
#
#    mean_rate = mean(FI_sw[t_stat_idx:end] ./ t_sw[t_stat_idx:end])
#    std_rate  = mean(err_sw[t_stat_idx:end] ./ t_sw[t_stat_idx:end])
#
#    println("  mean FI rate = $mean_rate  ($(round(time()-time_sw, digits=1)) s)")
#
#    # Append result immediately so partial runs are not lost
#    #open(csv_path, "a") do io
#    #    println(io, "$(k-1),$NS,$(Omega1/Omega_c),$dφ,$mean_rate,$std_rate")
#    #end
#    # Store time resolved data for the last single trajectory as well
#    writedlm("./FI_deltavarphi_sweep_N$(NS)_ratio$(Omega1/Omega_c)_deltavarphi$(dφ).dat",
#            hcat(t_sw, FI_sw, err_sw))
#    
#
#    @everywhere GC.gc(true)
#end
#println("Done: $csv_path")
@everywhere GC.gc(true)
rmprocs(workers())