using LinearAlgebra

export linsparse_GP, se_kernel, se_kernel_01, se_kernel_10, se_kernel_11, pre_compute, predict, linearize

mutable struct linsparse_GP
    rbf_variance
    rbf_lengthscale
    noise_variance
    lambda_inv
    X; Y; Z
    N; N_ind; n
    Sigma_inv; alpha; beta

    function linsparse_GP(d::Dict)
        rbf_variance = d["rbf_variance"]
        rbf_lengthscale = d["rbf_lengthscale"]
        noise_variance = d["noise_variance"]
        X = d["X_train"]
        Y = d["Y_train"]
        Z = d["X_ind"]

        N = size(Y,1)
        N_ind = size(Z,1)
        n = size(X,2)
        lambda_inv = Diagonal(rbf_lengthscale.^-2)

        new(rbf_variance, rbf_lengthscale, noise_variance, lambda_inv, X, Y, Z, N, N_ind, n)
    end
end

function se_kernel(m::linsparse_GP, x::AbstractVector, y::AbstractVector)
    return m.rbf_variance^2*exp(-0.5*transpose(x-y)*m.lambda_inv*(x-y))
end

function se_kernel_01(m::linsparse_GP, x::AbstractVector, y::AbstractVector)
    a = se_kernel(m, x, y)
    b = transpose(m.lambda_inv*(x-y))
    return a*b
end

function se_kernel_10(m::linsparse_GP, x::AbstractVector, y::AbstractVector)
    a = se_kernel(m, x, y)
    b = -m.lambda_inv*(x-y)
    return a*b
end

function se_kernel_11(m::linsparse_GP, x::AbstractVector, y::AbstractVector)
    l = size(x, 1)
    a = se_kernel(m, x, y)*Matrix(I, l, l)
    b = se_kernel_10(m, x, y)*transpose(x-y)
    return a + b
end

function pre_compute(m::linsparse_GP)
    # Pre-compute (K_N + sigma^2*I)^-1
    K_NZ = zeros(m.N, m.N_ind)
    K_ZZ = zeros(m.N_ind, m.N_ind)
    K_NN = zeros(m.N, m.N)
    for i = 1:m.N
        for j = 1:m.N_ind
            K_NZ[i,j] = se_kernel(m, m.X[i,:], m.Z[j,:])
        end
    end
    for i = 1:m.N_ind
        for j = 1:m.N_ind
            K_ZZ[i,j] = se_kernel(m, m.Z[i,:], m.Z[j,:])
        end
    end
    for i = 1:m.N
        for j = 1:m.N
            K_NN[i,j] = se_kernel(m, m.X[i,:], m.X[j,:])
        end
    end
    K_ZN = transpose(K_NZ)
    Q_NN = K_NZ*inv(K_ZZ)*K_ZN
    LAMBDA = diagm(diag(K_NN - Q_NN + m.noise_variance^2*Matrix(I, m.N, m.N)))
    m.Sigma_inv = inv(Q_NN + LAMBDA)
    m.alpha = inv(K_ZZ)*K_ZN*m.Sigma_inv*m.Y
    m.beta = inv(K_ZZ)*K_ZN*m.Sigma_inv*transpose(K_ZN)*transpose(inv(K_ZZ))
end

function predict(m::linsparse_GP, p::AbstractVector, use_var = false)
    K_star_Z = zeros(1, m.N_ind)
    for i = 1:m.N_ind
        K_star_Z[i] = se_kernel(m, p, m.Z[i,:])
    end
    mean = K_star_Z * m.alpha
    if use_var
        K_star = se_kernel(m, p, p)
        K_Z_star = transpose(K_star_Z)
        variance = K_star - K_star_Z*m.beta*K_Z_star + m.noise_variance^2
        return mean, variance
    else return mean
    end
end


function linearize(m::linsparse_GP, p::AbstractVector, use_var = false)
    K_star_Z = zeros(1, m.N_ind)
    K10_star_Z = zeros(m.n, m.N_ind)
    for i = 1:m.N_ind
        K_star_Z[i] = se_kernel(m, p, m.Z[i, :])
        K10_star_Z[:, i] = se_kernel_10(m, p, m.Z[i, :])
    end
    A = [K_star_Z; K10_star_Z]
    mx = A*m.alpha
    if use_var
        K_star = se_kernel(m, p, p)
        K01_star = se_kernel_01(m, p, p)
        K10_star = se_kernel_10(m, p, p)
        K11_star = se_kernel_11(m, p, p)
        B = [K_star K01_star; K10_star K11_star]
        C = A*m.beta*transpose(A)
        Vx = B-C
        return mx, Vx
    else return mx
    end
end
