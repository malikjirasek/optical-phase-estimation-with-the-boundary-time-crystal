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
