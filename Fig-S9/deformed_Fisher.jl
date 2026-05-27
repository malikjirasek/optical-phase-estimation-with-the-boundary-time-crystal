using LinearAlgebra
using SparseArrays
using DifferentialEquations
######################################### Methods #############################

"""
    deformed_Generic_Spin_Boson_model(NS, NB, Omega, dOmega, delta, lambda_TC, lambda_D, lambda_GD, omega_ph, kappa, timespan, init_state)

Constructs the deformed generator of the generic spin-boson model for sensing Omega.

# Arguments
    * `NS`:  Number of spins.
    * `NB`: Maximal occupation number of the cavity.
    * `Omega`: Driving of the two-level atoms.
    * `dOmega`: Small deviation of Omega for determining the numerical derivative.
    * `delta`: Detuning of the laser from the atom.
    * `lambda_TC`: Coupling of the atoms to the cavity. Excitation exchange mechanism.
    * `lambda_D`: Coupling of the atoms to the cavity. Dicke-like fashion.
    * `lambda_GD`: Coupling of the atoms to the cavity. Generalized Dicke-like fashion.
    * `omega_ph`: Detuning of the cavity from the laser.
    * `kappa`: Decay rate of the two-level atoms.
    * `timespan`: Timesteps, where the result is saved.
    * `init_state`: Initial state of the system.
# returns
    * `fisher_deformed`: System-environment QFI.
"""
function deformed_Generic_Spin_Boson_model(NS, NB, Omega, dOmega, delta, lambda_TC, lambda_D, lambda_GD, omega_ph, kappa, timespan, init_state)
    # Construct the Hamiltonian and the jump operators for the spin-boson model.

    # Built up collective spins.
    M=NS/2
    S=Int128(2*M+1)
    J=M*(M+1)
    b=-(-M:1:M)
    Sz=sparse(diagm(b))

    #Ladder operator collective spins.
    Sp=spzeros(Int128(S),Int128(S))
    for k1 in range(1,S)
        for k2 in range(1,S)
            m=Sz[k2,k2]
            if k1 == (k2-1)
                Sp[k1,k2]=sqrt((J)-m*(m+1))
            end
        end
    end
    Sx=sparse(1/2*(Sp+Sp'))
    Sy=sparse(1/2*(-1im*Sp+1im*Sp'))
    id_S=sparse(I,Int128(S),Int128(S))

    #Bosonic mode
    values_destroy = [sqrt(i) for i in range(1,NB-1)]
    destroy = spdiagm(1 => values_destroy)
    create = destroy'
    id_B = sparse(I,Int128(NB),Int128(NB))

    # Light-matter Hamiltonian
    H_int =  lambda_TC*(kron(destroy,Sp)+kron(create,Sp'))/sqrt(M) + lambda_D*kron(destroy+create,Sx)/sqrt(M) + lambda_GD*kron(destroy+create,Sz)/sqrt(M)
    # Bosonic mode 
    H_cavity = kron(omega_ph*(create*destroy), id_S)
    # Driving Hamiltonian
    H_spins_m = kron(id_B,(Omega-dOmega)*(Sp+Sp')*0.5)+ kron(id_B,delta*Sz)
    H_spins_p = kron(id_B,(Omega+dOmega)*(Sp+Sp')*0.5)+ kron(id_B,delta*Sz)
    # total Hamiltonian
    H_minus = H_spins_m + H_cavity + H_int
    H_plus = H_spins_p + H_cavity + H_int
    # Photon losses
    L = kron(sqrt(kappa)*destroy,id_S)
    #Identity 
    id = kron(id_B,id_S)

    #deformed_gen_mp
    deformed_gen_mp = sparse(-1im*kron(id,H_minus) + 1im*kron(transpose(H_plus),id) + kron(conj(L),L)-1/2*kron(id, L'*L)-1/2*kron(transpose(L'*L),id))
    #deformed_gen_pm
    deformed_gen_pm = sparse(-1im*kron(id,H_plus) + 1im*kron(transpose(H_minus),id) + kron(conj(L),L)-1/2*kron(id, L'*L)-1/2*kron(transpose(L'*L),id))

    #init
    dim = size(H_plus)[1]
    rho_0 = init_state*init_state'
    rho_0_vec = vec(rho_0)

    #Integrate equation
    function deformed_mp(du, u, p, t)
        du .= deformed_gen_mp*u
    end
    function deformed_pm(du, u, p, t)
        du .=  deformed_gen_pm*u
    end
    tspan = (timespan[1], timespan[end])
    prob_mp = ODEProblem(deformed_mp, rho_0_vec, tspan)
    sol_mp = solve(prob_mp, Tsit5(),  abstol = 1e-10, reltol= 1e-10,saveat=timespan)
    prob_pm = ODEProblem(deformed_pm, rho_0_vec, tspan)
    sol_pm = solve(prob_pm, Tsit5(),  abstol = 1e-10, reltol= 1e-10,saveat=timespan)

    fisher_deformed = zeros(length(timespan))
    for (t_idx, t)  in enumerate(timespan)
        tr_mp = tr(reshape(sol_mp.u[t_idx], (dim, dim)))
        tr_pm = tr(reshape(sol_pm.u[t_idx], (dim, dim)))
        fisher_deformed[t_idx] = 4*real(-(log(tr_mp)+log(tr_pm))/(4*dOmega*dOmega))
    end
    return fisher_deformed
end

"""
    deformed_SB_Dicke_lambda( NS, NB, Omega, delta, lambda, dlambda, kappa, eta, timespan, init_state )

Constructs the deformed generator of the generalized Dicke model for sensing lambda.

# Arguments
    * `NS`:  Number of spins.
    * `NB`: Maximal occupation number of the cavity.
    * `Omega`: Driving of the two-level atoms.
    * `lambda`: Coupling of the atoms to the cavity. Generalized-Dicke like.
    * `dlambda`: Small deviation of the coupling for numerical derivative.
    * `kappa`: Decay rate of the cavity.
    * `eta`: Detuning of the cavity from the laser.
    * `timespan`: Timespan for the evolution.
    * `init_state`: Initial state of the system.
# returns
    * `fisher_deformed`: System-environment QFI.
"""
function deformed_SB_Dicke_lambda( NS, NB, Omega, delta, lambda, dlambda, kappa, eta, timespan, init_state )
    # Construct the Hamiltonian and the jump operators for the spin-boson model.

    # Built up collective spins.
    M=NS/2
    S=Int128(2*M+1)
    J=M*(M+1)
    b=-(-M:1:M)
    Sz=sparse(diagm(b))

    #Ladder operator collective spins.
    Sp=spzeros(Int128(S),Int128(S))
    for k1 in range(1,S)
        for k2 in range(1,S)
            m=Sz[k2,k2]
            if k1 == (k2-1)
                Sp[k1,k2]=sqrt((J)-m*(m+1))
            end
        end
    end
    id_S=sparse(I,Int128(S),Int128(S))

    #Bosonic mode
    values_destroy = [sqrt(i) for i in range(1,NB-1)]
    destroy = spdiagm(1 => values_destroy)
    create = destroy'
    id_B = sparse(I,Int128(NB),Int128(NB))

    # Light-matter Hamiltonian
    H_int_m =  (lambda-dlambda)*(kron(destroy+create,Sz))/sqrt(M)
    H_int_p =  (lambda+dlambda)*(kron(destroy+create,Sz))/sqrt(M)
    # Driving Hamiltonian
    H_spins = kron(id_B,Omega*(Sp+Sp')*0.5)+ kron(id_B,delta*Sz)
    # Bosonic mode 
    H_cavity = kron(eta*(create*destroy), id_S)

    H_minus = H_spins + H_cavity + H_int_m
    H_plus = H_spins + H_cavity + H_int_p

    # Photon losses
    L = kron(sqrt(kappa)*destroy,id_S)

    #Identity 
    id = kron(id_B,id_S)

    #deformed_gen_mp
    deformed_gen_mp = sparse(-1im*kron(id,H_minus) + 1im*kron(transpose(H_plus),id) + kron(conj(L),L)-1/2*kron(id, L'*L)-1/2*kron(transpose(L'*L),id))
    #deformed_gen_pm
    deformed_gen_pm = sparse(-1im*kron(id,H_plus) + 1im*kron(transpose(H_minus),id) + kron(conj(L),L)-1/2*kron(id, L'*L)-1/2*kron(transpose(L'*L),id))

    #init
    dim = size(H_plus)[1]
    rho_0 = init_state*init_state'
    rho_0_vec = vec(rho_0)
    #Integrate equation
    function deformed_mp(du, u, p, t)
        du .= deformed_gen_mp*u
    end
    function deformed_pm(du, u, p, t)
        du .=  deformed_gen_pm*u
    end
    tspan = (timespan[1], timespan[end])
    prob_mp = ODEProblem(deformed_mp, rho_0_vec, tspan)
    sol_mp = solve(prob_mp, Tsit5(), abstol = 1e-10, reltol= 1e-10,saveat=timespan)
    prob_pm = ODEProblem(deformed_pm, rho_0_vec, tspan)
    sol_pm = solve(prob_pm, Tsit5(), abstol = 1e-10, reltol= 1e-10, saveat=timespan)


    fisher_deformed = zeros(length(timespan))
    for (t_idx, t)  in enumerate(timespan)
        tr_mp = tr(reshape(sol_mp.u[t_idx], (dim, dim)))
        tr_pm = tr(reshape(sol_pm.u[t_idx], (dim, dim)))
        fisher_deformed[t_idx] = 4*real(-(log(tr_mp)+log(tr_pm))/(4*dlambda*dlambda))
    end
    return fisher_deformed
end