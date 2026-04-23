using LinearAlgebra
using SparseArrays
#######################################HelperFunctions########################################
#Creates the Dicke states 
function Dicke_state(theta, phi, J)
    psi0 = zeros(Complex{Float64}, Int128(2*J+1))
    idx = 1
    for M in (-J:1:J)
        psi0[idx] = sqrt(binomial(Int128(2*J),Int128(J+M)))*(cos(theta/2))^(J-M)*(sin(theta/2))^(J+M)*exp(-1im*(J+M)*phi)
        idx += 1
    end
    return psi0/norm(psi0)
end
####################################### Models ################################
"""
    Generic_Spin_Boson_model(NS, NB, Omega, delta, lambda_TC, lambda_D, omega_ph, kappa)

Generates the Hamiltonian and the jump operators of the spin-boson model.

# Arguments
    * `NS`:  Number of spins.
    * `NB`: Maximal occupation number of the cavity.
    * `Omega`: Driving of the two-level atoms.
    * `delta`: Detuning of the laser from the atom.
    * `lambda_TC`: Coupling of the atoms to the cavity. Excitation exchange mechanism.
    * `lambda_D`: Coupling of the atoms to the cavity. Dicke-like fashion.
    * `lambda_GD`: Coupling of the atoms to the cavity. Generalized Dicke-like fashion.
    * `omega_ph`: Detuning of the cavity from the laser.
    * `kappa`: Decay rate of the cavity.
# returns
    * `H`: Hamiltonian of the system.
    * `Hdomega`: Derivative of the Hamiltonian with repsect to the drive of the atoms.
    * `Hdlambda_TC`: Derivative of the Hamiltonian with repsect to the Tavis-Cummings coupling.
    * `Hdlambda_D`: Derivative of the Hamiltonian with repsect to the Dicke coupling.
    * `Hdlambda_GD:  Derivative of the Hamiltonian with repsect to the generalized Dicke coupling.
    * `L`: Jump operators for the photon losses from the cavity.
    * `(operators)`: List of useful operators.
    * `(identities)`: Identities of the collective spin and the bosonic mode.
"""

function Generic_Spin_Boson_model(NS, NB, Omega, delta, lambda_TC, lambda_D, lambda_GD, omega_ph, kappa)
    # Construct the Hamiltonian and the jump operator of the spin-boson model.

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
    # Driving Hamiltonian
    H_spins = kron(id_B,Omega*(Sp+Sp')*0.5) + kron(id_B,delta*Sz)
    # Bosonic mode 
    H_cavity = kron(omega_ph*(create*destroy), id_S)

    # total Hamiltonian
    H = H_spins + H_cavity + H_int
    # Derivative of the Hamiltonian with respect to the different parameters
    Hdomega = kron(id_B,(Sp+Sp')*0.5)
    Hdlambda_TC = (kron(destroy,Sp)+kron(create,Sp'))/sqrt(M)
    Hdlambda_D = kron(destroy+create,Sx)/sqrt(M)
    Hdlambda_GD = kron(destroy+create,Sz)/sqrt(M)
    # Photon losses
    L = kron(sqrt(kappa)*destroy,id_S)
    dL =0

    # Expectation values 

    X = kron(id_B, Sx)
    Y = kron(id_B, Sy)
    Z = kron(id_B, Sz)
    n = kron(create*destroy,id_S)
    return (H, Hdomega, Hdlambda_TC, Hdlambda_D, Hdlambda_GD, L, dL, (X, Y, Z, n), ( id_B, id_S) )
end


function BTC_model(NS, Omega, kappa, varphi)
    # Construct the Hamiltonian and the jump operator of the spin-boson model.

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

    # Driving Hamiltonian
    H = 0.5*Omega*(Sp+Sp')

    # Derivative of the Hamiltonian with respect to the different parameters
    Hdomega = 0.5*(Sp+Sp')
    Hdvarphi = 0

    # Photon losses
    L = sqrt(kappa)*(Sp')*exp(-1im*varphi)
    dLomega = 0
    dLvarphi = -1im*sqrt(kappa)*(Sp')*exp(-1im*varphi)

    # Expectation values 

    X = Sx
    Y = Sy
    Z = Sz
    return (H, Hdomega, Hdvarphi, L, dLomega, dLvarphi, (X, Y, Z), ( id_S) )
end


function BTC_cascaded_model(NS, Omega1, Omega2, kappa, deltavarphi)
    # Construct the Hamiltonian and the jump operator of the spin-boson model.

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

    # Hamiltonian
    Hcasc = -1im*(kappa/2.0)*(kron(Sp,Sp')*exp(-1im*deltavarphi)-kron(Sp',Sp)*exp(1im*deltavarphi))
    H = 0.5*Omega1*(kron(Sp,id_S)+kron(Sp',id_S)) + 0.5*Omega2*(kron(id_S,Sp)+kron(id_S,Sp')) + Hcasc

    # Derivative of the Hamiltonian with respect to the different parameters
    Hdomega = 0.5*(kron(Sp,id_S)+kron(Sp',id_S))
    Hdvarphi = -1*(kappa/2.0)*(kron(Sp',Sp)*exp(-1im*deltavarphi)-kron(Sp,Sp')*exp(1im*deltavarphi))

    # Photon losses
    L = sqrt(kappa)*(kron(Sp',id_S)*exp(-1im*deltavarphi)+kron(id_S,Sp'))
    dLomega = 0
    dLvarphi = -1im*sqrt(kappa)*kron(Sp',id_S)*exp(-1im*deltavarphi)

    # Expectation values 

    X1 = kron(Sx,id_S)
    Y1 = kron(Sy,id_S)
    Z1 = kron(Sz,id_S)
    return (H, Hdomega, Hdvarphi, L, dLomega, dLvarphi, (X1, Y1, Z1), (kron(id_S,id_S)) )
end

