using Distributed
using LinearAlgebra
using SparseArrays 
using FastExpm
######################################### Methods #############################


"""
    Fisher_trajectory_homodyne( H, dH, L, dL, num_traj, psi0, ops, t_final, dt, phase )

Implementation of the quantum jump trajectories of a continuously monitored quantum system with ideal homodyne detection.

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
    * `phase`: Phase of the local strong oscillator used for homodyne detection.

#returns 
    #Different relevant quantities averaged over the trajectories.
    * `t`: Evaluated times.
    * `expectation_values[1,:]`: Expectation value of ops[1].
    * `expectation_values[4,:]`: Expectation value of ops[4].
    * `current`: Homodyne current.
    * `FisherT`: FI for different times.
    * `QFisherT`: QFI for different times.
    * `real_part_overlap`: Real part of the inner product given in Eq.(8) (II).
    * `imag_part_overlap`: Imag part of the inner product given in Eq.(8) (II).
"""

function Fisher_trajectory_homodyne_jump( H, L, dL, num_traj, psi0, ops, t_final, dt, phase )
    # Calculate the trajectories for the given Hamiltonian and the jump operator.

    # Calculate Wiener increment
    dW() = sqrt(dt) * randn()
    
    # Create timespan.
    Ntime = Int(floor(t_final/dt)) # Number of timesteps
    t = (1 : Ntime) * dt

    # Initialize expectation values
    num_ops = length(ops)
    expectation_values = zeros((num_ops, Ntime))

    # Time record of the homodyne current
    current = zero(t)

    last_state = psi0

    # Average over several trajectories.
    result = @distributed (+)for ktraj = 1 : num_traj
        psi = copy(psi0) # Assign initial state to each trajectory
        normalized_psi = psi
        # Derivative of the state with respect to the parameter
        dpsi = zero(psi)
        phi = zero(psi)

        # Initialize quantities of interest
        FisherT = similar(t)
        QFisherT = similar(t)
        real_part_overlap = zero(t)
        imag_part_overlap = zero(t)

        for jt=1:Ntime
            # Determine the homodyne current
            wiener_num=dW() 
            dJ = real(normalized_psi'*(exp(1im*phase)*L+exp(-1im*phase)*L')*normalized_psi)*dt+wiener_num
            current[jt] = dJ

            # Kraus operator associated to the homodyne current
            M = I -1im*H*dt - 0.5*(L' * L)*dt + exp(1im*phase)*L*dJ


            # Derivative of the Kraus-like operator with respect to the parameter
            # The second part should not be there (benchmarked against analytically known case)
            dM = exp(1im*phase)*dL*dJ #+ exp(1im*phase)*L*(real(normalized_psi'*(exp(1im*phase)*dL+exp(-1im*phase)*dL')*normalized_psi)*dt+wiener_num)
            
            # Evolving the state with the Kraus operator
            new_psi = M * normalized_psi
            norm_psi = sqrt(new_psi'*new_psi)

            # Evolving the phi
            phi = (dM*normalized_psi+M*phi) / norm_psi

            # Renormalize psi
            normalized_psi = new_psi/norm_psi
            
            real_part_overlap[jt] = real(normalized_psi' * phi)
            imag_part_overlap[jt] = imag(normalized_psi' * phi)

            last_state = normalized_psi

            # Determine the derivative with respect to the parameter
            dpsi = phi - 0.5 * (phi' * normalized_psi + normalized_psi' * phi) * normalized_psi;

            # Classical FI for the continuous measurement (homodyne current)
            FisherT[jt] = real((normalized_psi' * phi + phi' * normalized_psi)^2)

            # QFI of the system at time t
            QFisherT[jt] = 4 * real( dpsi' * dpsi + (dpsi' * normalized_psi)^2)

            # Expectation values
            for i in range(1,num_ops)
                expectation_values[i,jt] = real(normalized_psi'*ops[i]*normalized_psi)
            end
        end

        hcat(expectation_values[2,:], expectation_values[3,:], current, FisherT, QFisherT, real_part_overlap, imag_part_overlap)
    end
    return (t, result[:,1] ./ num_traj, result[:,2] ./ num_traj, result[:,3] ./ num_traj, result[:,4] ./ num_traj, result[:,5] ./ num_traj, result[:,6] ./ num_traj, result[:,7] ./ num_traj)
end


#### Optimized with input from Copilot.
function Fisher_trajectory_homodyne_jump_optimized(
    H, L, dL, num_traj, psi0, ops, t_final, dt, phase, num_steps_save
)
    # Wiener increment
    dW() = sqrt(dt) * randn()

    # Full number of timesteps
    Ntime_full = Int(floor(t_final/dt))

    # Number of saved points
    Nsave = ceil(Int, Ntime_full / num_steps_save)

    # Time vector for saved points
    t = (1:Nsave) .* (dt * num_steps_save)

    num_ops = length(ops)

    # Arrays for saved expectation values
    expectation_values = zeros(num_ops, Nsave)
    current = zeros(Nsave)

    last_state = psi0

    result = @distributed (+) for ktraj = 1:num_traj
        psi = copy(psi0)
        normalized_psi = psi
        dpsi = zero(psi)
        phi = zero(psi)

        FisherT = zeros(Nsave)
        FisherT2 = zeros(Nsave)
        #QFisherT = zeros(Nsave)
        #real_part_overlap = zeros(Nsave)
        #imag_part_overlap = zeros(Nsave)

        save_index = 1


        # psi0 :: Vector{ComplexF64}
        dim = length(psi0)

        M      = similar(L)                 # Complex sparse matrix
        dM     = similar(L)                 # Complex sparse matrix
        new_psi = similar(psi0)             # Complex vector
        tmp_vec = similar(psi0)             # Complex vector

        # Deterministic part of M (complex)
        M0 = spdiagm(0 => ones(dim)) .- 1im * H * dt .- 0.5 * (L' * L) * dt


        @inbounds @fastmath for jt = 1:Ntime_full
            # Wiener increment
            wiener_num = sqrt(dt) * randn()
            dJ = real(normalized_psi' * (exp(1im*phase)*L + exp(-1im*phase)*L') * normalized_psi) * dt + wiener_num

            # M = M0 + exp(iφ)*L*dJ  (all complex, sparse)
            copyto!(M, M0)
            # M .+= exp(1im*phase) * L * dJ  is not valid for sparse; use axpy!
            # M = M0 + α*L  → M .= M0; M .+= α*L is not defined for sparse, so:
            M .= M0 + (exp(1im*phase) * dJ) .* L  # or use a custom sparse combination if needed

            # dM = exp(iφ)*dL*dJ
            dM .= (exp(1im*phase) * dJ) .* dL

            # new_psi = M * normalized_psi
            mul!(new_psi, M, normalized_psi)

            norm_psi = sqrt(real(new_psi' * new_psi))

            # tmp_vec = M * phi
            mul!(tmp_vec, M, phi)

            # phi = (dM*psi + tmp_vec) / norm_psi
            mul!(phi, dM, normalized_psi)   # phi = dM*psi
            @. phi = (phi + tmp_vec) / norm_psi

            @. normalized_psi = new_psi / norm_psi

            rp = real(normalized_psi' * phi)
            ip = imag(normalized_psi' * phi)

            overlap = 0.5 * (phi' * normalized_psi + normalized_psi' * phi)
            @. dpsi = phi - overlap * normalized_psi

            # ... saving logic as before ...
            # Save only every num_steps_save
            if jt % num_steps_save == 0
                FisherT[save_index] = real((normalized_psi' * phi + phi' * normalized_psi)^2)
                FisherT2[save_index] = real((normalized_psi' * phi + phi' * normalized_psi)^2)^2
                #QFisherT[save_index] = 4 * real(dpsi' * dpsi + (dpsi' * normalized_psi)^2)
                #real_part_overlap[save_index] = rp
                #imag_part_overlap[save_index] = ip
                current[save_index] = dJ

                for i in 1:num_ops
                    expectation_values[i, save_index] = real(normalized_psi' * ops[i] * normalized_psi)
                end

                save_index += 1
            end

            last_state = normalized_psi
        end

        hcat(expectation_values[2,:], expectation_values[3,:], current,
             FisherT,FisherT2)# QFisherT, real_part_overlap, imag_part_overlap)
    end
    
    meanFisher = result[:,4] ./ num_traj
    meanFisher2 = result[:,5] ./ num_traj

    varFisher = meanFisher2 .- meanFisher.^2
    stdFisher = sqrt.(varFisher)./sqrt(num_traj-1)
    

    return (
        t,
        result[:,1] ./ num_traj,
        result[:,2] ./ num_traj,
        result[:,3] ./ num_traj,
        meanFisher,
        stdFisher
        )
        #result[:,5] ./ num_traj,
        #result[:,6] ./ num_traj,
        #result[:,7] ./ num_traj
    #)
end
