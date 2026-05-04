using Distributed
using LinearAlgebra
using SparseArrays 
using FastExpm
using ProgressMeter
######################################### Methods #############################
"""
    Fisher_trajectory( H, dH, L, dL, num_traj, psi0, ops, t_final, dt )

Implementation of the quantum jump trajectories of a continuously monitored quantum system with ideal photo-counting.

Evolution for pure states.

# Arguments
    * `H`:  Hamiltonian of the system.
    * `dH`:  Derivative of the Hamiltonian of the system with respect to the parameter of interest.
    * `L`: Jump operator.
    * `dL`: Derivative of the Jump operator.
    * `num_traj`: Number of trajectories.
    * `psi0`: Initial state for the evolution.
    * `ops`:  List of operators for which the expectation value will be 
    calculated.
    * `t_final`: Final time for the evolution.
    * `dt`: Infinitesimal time step for the evolution.

#returns 
    #Different relevant quantities averaged over the trajectories.
    * `t`: Evaluated times.
    * `expectation_values[1,:]`: Expectation value of ops[1].
    * `expectation_values[4,:]`: Expectation value of ops[4].
    * `emissions`: Emission record.
    * `FisherT`: FI for different times.
    * `QFisherT`: QFI for different times.
    * `real_part_overlap`: Real part of the inner product given in Eq.(8) (II).
    * `imag_part_overlap`: Imag part of the inner product given in Eq.(8) (II).
"""

function Fisher_trajectory( H, dH, L, dL, num_traj, psi0, ops, t_final, dt )
    # Calculate the trajectories for the given Hamiltonian and the jump operator.

    # Construct effective Hamiltonian
    H_eff = H - 0.5im * (L' * L)

    # Exact `free` evolution in Kraus operator form
    M0 = sparse(fastExpm(- 1im * H_eff *dt ))
    # Kraus operator associated with the jumps
    M1 = sqrt(dt) * L


    # Derivative of the Kraus-like operator with respect to the parameter
    dHeff = dH - 0.5im * (dL' * L + L' * dL)
    dM0 = (- 1im * dHeff * dt) * M0
    dM1 = sqrt(dt) * dL
    
    # Create timespan
    Ntime = Int(floor(t_final/dt)) # Number of timesteps
    t = (1 : Ntime) * dt

    # Initialize expectation values
    num_ops = length(ops)
    expectation_values = zeros((num_ops, Ntime))

    # Time record of the emissions
    emissions = zero(t)


    # Average over several trajectories.
    result = @distributed (+)for ktraj = 1 : num_traj
        psi = copy(psi0) # Assign initial state to each trajectory
        normalized_psi = psi
        # Derivative of the state with respect to the parameter
        dpsi = zero(psi)
        phi = zero(psi)

        # Initialize quantities of interest
        FisherT = zero(t)
        QFisherT = zero(t)
        real_part_overlap = zero(t)
        imag_part_overlap = zero(t)

        # Create a random number between 0 and 1
        jump_prob = rand() 
        for jt=1:Ntime
            # Inner product of the unnormalized state
            # The inner product decreases during the evolution with the effective Hamiltonian
            pPD = psi'*psi
            # Has the photon been detected?
            # A jump occurs if the inner product is smaller than the random number 
            if (jump_prob > real(pPD)) # Detected

                # Save emission in protocol
                emissions[jt] = 1

                # Apply the jump operator
                new_psi = M1 * psi

                # Normalize the state
                norm_psi = sqrt(new_psi'*new_psi)
                psi = new_psi / norm_psi
                # Normalized updated state
                updated_normalized_psi = M1 * normalized_psi
                norm_psi_normalized = sqrt(updated_normalized_psi'*updated_normalized_psi)

                # Update the derivative with the previous normalized state
                phi = (dM1 * normalized_psi + M1 * phi) / norm_psi_normalized

                # Generate a new random number
                jump_prob = rand()
    
            else # Not detected
                
                # "Free" evolution with the effective Hamiltonian
                new_psi = M0 * psi

                # Do not normalize the state
                psi=new_psi
                # Save norm 
                norm_psi = sqrt(new_psi'*new_psi)
                # Update normalized state
                updated_normalized_psi = M0 * normalized_psi
                norm_psi_normalized = sqrt(updated_normalized_psi'*updated_normalized_psi)
                
                # Update the derivative with the previous normalized state
                phi = (dM0 * normalized_psi + M0 * phi) / norm_psi_normalized 

            end

            # Renormalize psi
            normalized_psi = new_psi/norm_psi
            
            real_part_overlap[jt] = real(normalized_psi' * phi)
            imag_part_overlap[jt] = imag(normalized_psi' * phi)

            # Determine the derivative with respect to the parameter
            dpsi = phi - 0.5 * (phi' * normalized_psi + normalized_psi' * phi) * normalized_psi

            # Classical FI for the continuous measurement (emission record)
            FisherT[jt] = real((normalized_psi' * phi + phi' * normalized_psi)^2)

            # QFI of the system at time t
            QFisherT[jt] = 4 * real( dpsi' * dpsi + (dpsi' * normalized_psi)^2)

            # Expectation values
            for i in range(1,num_ops)
                expectation_values[i,jt] = real(normalized_psi'*ops[i]*normalized_psi)
            end
        end

        hcat(expectation_values[2,:], expectation_values[3,:], emissions, FisherT, QFisherT, real_part_overlap, imag_part_overlap)
    end
    return (t, result[:,1] ./ num_traj, result[:,2] ./ num_traj, result[:,3] ./ num_traj, result[:,4] ./ num_traj, result[:,5] ./ num_traj, result[:,6] ./ num_traj, result[:,7] ./ num_traj)
end

function Fisher_trajectory_reduced(H, dH, L, dL, num_traj, psi0, ops,
                                   t_final, dt, n_save_steps)

    dim   = length(psi0)
    H_eff = H - 0.5im * (L' * L)
    # Dense matrices are faster than sparse for dim~49 since fill-in is high
    M0  = Matrix(fastExpm(-1im * H_eff * dt))
    M1  = Matrix(sqrt(dt) * L)

    dHeff = dH - 0.5im * (dL' * L + L' * dL)
    dM0   = Matrix((-1im * dHeff * dt) * M0)
    dM1   = dL isa AbstractMatrix ? Matrix(sqrt(dt) * dL) :
                                    zeros(ComplexF64, dim, dim)

    # Pre-convert observable matrices to dense once
    op2 = Matrix(ops[2])
    op3 = Matrix(ops[3])

    Ntime = Int(floor(t_final / dt))
    Nsave = Int(floor(Ntime / n_save_steps))

    result = @showprogress 1 "Computing trajectories..." @distributed (+) for ktraj = 1:num_traj

        psi            = copy(psi0)
        normalized_psi = copy(psi0)
        phi            = zeros(ComplexF64, dim)
        # Three scratch buffers reused every step — no per-step allocation
        buf_a = Vector{ComplexF64}(undef, dim)  # M * psi  /  M * normalized_psi
        buf_b = Vector{ComplexF64}(undef, dim)  # M * normalized_psi (norm branch)
        buf_c = Vector{ComplexF64}(undef, dim)  # M * phi (old)

        exp2      = zeros(Nsave)
        exp3      = zeros(Nsave)
        emissions = zeros(Nsave)
        FisherT   = zeros(Nsave)
        FisherT2  = zeros(Nsave)

        jump_prob = rand()
        save_idx  = 1

        @inbounds for jt = 1:Ntime

            pPD = real(dot(psi, psi))

            if jump_prob > pPD  # photon detected — jump branch
                mul!(buf_a, M1, psi)
                norm_psi = norm(buf_a)
                psi .= buf_a ./ norm_psi

                mul!(buf_b, M1, normalized_psi)
                norm_psi_norm = norm(buf_b)

                # phi = (dM1*normalized_psi + M1*phi_old) / norm_psi_norm
                mul!(buf_c, M1, phi)
                mul!(phi, dM1, normalized_psi)
                @. phi = (phi + buf_c) / norm_psi_norm

                normalized_psi .= buf_b ./ norm_psi_norm
                jump_prob     = rand()
                emission_flag = 1.0
            else  # no-jump branch
                mul!(buf_a, M0, psi)
                norm_psi = norm(buf_a)

                mul!(buf_b, M0, normalized_psi)
                norm_psi_norm = norm(buf_b)

                # phi = (dM0*normalized_psi + M0*phi_old) / norm_psi_norm
                mul!(buf_c, M0, phi)
                mul!(phi, dM0, normalized_psi)
                @. phi = (phi + buf_c) / norm_psi_norm

                normalized_psi .= buf_b ./ norm_psi_norm
                psi .= buf_a
                emission_flag = 0.0
            end

            if jt % n_save_steps == 0
                score = 2 * real(dot(normalized_psi, phi))
                mul!(buf_a, op2, normalized_psi)
                exp2[save_idx]      = real(dot(normalized_psi, buf_a))
                mul!(buf_a, op3, normalized_psi)
                exp3[save_idx]      = real(dot(normalized_psi, buf_a))
                emissions[save_idx] = emission_flag
                FisherT[save_idx]   = score^2
                FisherT2[save_idx]  = score^4
                save_idx += 1
            end
        end

        hcat(exp2, exp3, emissions, FisherT, FisherT2)
    end

    t_saved       = (n_save_steps:n_save_steps:Ntime) .* dt
    mean_exp2     = result[:,1] ./ num_traj
    mean_exp3     = result[:,2] ./ num_traj
    mean_emission = result[:,3] ./ num_traj
    mean_FisherT  = result[:,4] ./ num_traj
    mean_FisherT2 = result[:,5] ./ num_traj

    var_FisherT = mean_FisherT2 .- mean_FisherT.^2
    if any(var_FisherT .< -1e-12)
        @warn "Negative variance exceeded threshold at some time points."
    end
    var_FisherT = max.(var_FisherT, 0.0)
    err_FisherT = sqrt.(var_FisherT ./ (num_traj - 1))

    return (t_saved, mean_exp2, mean_exp3, mean_emission, mean_FisherT, err_FisherT)
end


