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
NS         = 6
kappa      = 1
Omega_c    = kappa * NS / 2
dt         = 0.001
t_final    = 3e4
n_save     = 500           # save every n_save steps  →  dt_save = 0.5
num_traj   = 27

# Steady-state averaging window for the FI rate
t_stat     = 1e4
t_stat_idx = Int(floor(t_stat / (n_save * dt)))

# Initial state: |J,J> ⊗ |J,J>
theta = 0
phi   = 0
init_state_spins = Dicke_state(theta, phi, NS/2)
init_state_spins = init_state_spins / norm(init_state_spins)
psi0_SB          = kron(init_state_spins, init_state_spins)

# Sweep grid
ratios             = [0.25]#, 0.25, 0.75, 4.0]
deltavarphi_values = exp10.(range(log10(0.001), log10(0.1), length=2))

println("dt = $dt   t_final = $t_final   n_save = $n_save   num_traj = $num_traj")
println("Steady-state window: t ∈ [$t_stat, $t_final]   (saved index $t_stat_idx:end)")
println("Sweep: $(length(ratios)) ratios × $(length(deltavarphi_values)) deltavarphi = $(length(ratios)*length(deltavarphi_values)) runs")

# ---------- legacy single-parameter time-resolved run (kept for reference) ----
#H, Hdomega, Hdvarphi, L, dLomega, dLvarphi, expectation_values, _ = BTC_cascaded_model(NS, Omega1, Omega2, kappa, deltavarphi)
#t_SB, exp_val1_SB, exp_val2_SB, emissions_SB, FisherT_SB_omega, ErrorFisher = Fisher_trajectory_reduced( H, Hdvarphi, L, dLvarphi, num_traj, psi0_SB, ops_SB, t_final, dt, n_save)
#writedlm("./FI_time_resolved_N$(NS)_ratio$(Omega1/Omega_c).dat", hcat(t_SB, FisherT_SB_omega, ErrorFisher))

############################################ Sweep ############################
# Output directory structure (anchored at the script's directory):
#   <script_dir>/sweep_dt<dt>_<num_traj>trajs/
#       ├── summary.csv                           (one row per (ratio, dφ))
#       └── ratio_<r>/
#             └── time_resolved_deltavarphi_<dφ>.dat
sweep_dir   = joinpath(@__DIR__, "sweep_dt$(dt)_$(num_traj)trajs")
mkpath(sweep_dir)
summary_csv = joinpath(sweep_dir, "summary.csv")
open(summary_csv, "w") do io
    println(io, ",NS,ratio,deltavarphi,mean_FI_rate,std_FI_rate")
end
println("Output directory: $sweep_dir")
println("Summary CSV:      $summary_csv")

row_idx     = 0
total_runs  = length(ratios) * length(deltavarphi_values)
total_start = time()

for (i, ratio) in enumerate(ratios)
    Omega1 = ratio * Omega_c
    Omega2 = Omega1
    ratio_dir = joinpath(sweep_dir, "ratio_$(ratio)")
    mkpath(ratio_dir)
    println("\n========== [$i/$(length(ratios))] ratio = $ratio   (Omega1 = $Omega1) ==========")

    for (j, dφ) in enumerate(deltavarphi_values)
        run_idx = (i - 1) * length(deltavarphi_values) + j
        println("[$run_idx/$total_runs] ratio = $ratio,  deltavarphi = $dφ")
        t0 = time()

        H_sw, _, Hdvarphi_sw, L_sw, _, dLvarphi_sw, ops_sw, _ =
            BTC_cascaded_model(NS, Omega1, Omega2, kappa, dφ)

        t_sw, _, _, _, FI_sw, err_sw =
            Fisher_trajectory_reduced(H_sw, Hdvarphi_sw, L_sw, dLvarphi_sw,
                                       num_traj, psi0_SB, ops_sw,
                                       t_final, dt, n_save)

        mean_rate = mean(FI_sw[t_stat_idx:end]  ./ t_sw[t_stat_idx:end])
        std_rate  = mean(err_sw[t_stat_idx:end] ./ t_sw[t_stat_idx:end])

        println("  mean FI rate = $mean_rate  ±  $std_rate     ($(round(time()-t0, digits=1)) s)")

        # Append to consolidated summary CSV (incremental — survives crashes)
        open(summary_csv, "a") do io
            println(io, "$row_idx,$NS,$ratio,$dφ,$mean_rate,$std_rate")
        end
        global row_idx += 1

        # Time-resolved data: one file per (ratio, deltavarphi), grouped by ratio
        time_resolved_path =
            joinpath(ratio_dir, "time_resolved_deltavarphi_$(dφ).dat")
        writedlm(time_resolved_path, hcat(t_sw, FI_sw, err_sw))

        @everywhere GC.gc(true)
    end
end

println("\nDone. Total wall time: $(round(time() - total_start, digits=1)) s")
println("Output directory: $sweep_dir")
@everywhere GC.gc(true)
rmprocs(workers())
