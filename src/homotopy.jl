# CREDITS : Nicolas Nadisic
# Download Giant Library at https://gitlab.com/nnadisic/giant.jl

# using Distances: pairwise,Euclidean     # Usage : knn_homotopy_solver - broadcasted_knn_homotopy_solver
# using LinearAlgebra: pinv               # Usage : homotopy (pinv) - broadcasted_homotopy (pinv)

# include("utils.jl")
"""
    homotopy(A,b,metric,[tol,maxIter,verbose,acc]) -> x [::Vector{acc}]

    This function gives the optimal solution x of the problem
        min_x ||Ax-b||_F + lambda ||x||_1
    that meet the given criterion metric

    INPUT :
    A = Data matrix [::Matrix]
    b = Target solution [::Vector]
    metric = function used to evaluate quality of the solution 
        and stop homotopy if solution is optimal [::Function metric(A,b,x) -> ::Bool]
    
    OPTIONAL :
    tol = Threshold used to stop homotopy based on metric function results [::Real]
    maxIter = Maximum number of iterations of ADMM [::Integer]
    verbose = If true, print lambda and sol on each iteration [::Bool]
    acc = Computing accuracy (type of inner variables, defaut is Float32) [::Float subtype]

    OUTPUT :
    solution = First solution to meet metric criterion [::Vector{acc}]
    
    -------------------------------------------------------------------------
Copyright @ Nicolas Nadisic, 2022 & Florian Dubois, 2024
"""
function homotopy(A::AbstractArray{<:Real,2},
                  b::AbstractVector{<:Real},
                  metric::Function;
                  tol::Float64 = 1e-12,
                  maxIter::Integer = 1000,
                  verbose::Bool=false,
                  acc::Type = Float32)

    # Debug tools
    # if !(tol >0); error("Negative tollerance ! Try @assert tol > 0") end
    # if !(acc <: Float); error("Accuracy must be a Float ! Try @assert acc <: Float) end
    # if !(maxIter > 0); error("Negative maxIter ! Try @assert maxIter > 0") end

    # Init 
    N = size(A, 2)
    lambda = zero(acc)

    AtA = Matrix{acc}(undef,N,N)
    Atb = A'b

    # Pre-allocate Containers
    K = falses(N) # active set (initial support is empty)
    theta = Vector{acc}(undef,N) # gradient vector of l1-norm
    v = Vector{acc}(undef,N) # Container for lambda candidates
    a1, a2, a3, a4 = zeros(acc,N),Vector{acc}(undef,N),Vector{acc}(undef,N),Vector{acc}(undef,N) # Container for b*, Δb, Ar* and sign(Ar*) - Δr
    q1,q2 = BitArray(undef,N),BitArray(undef,N) # Mask for feature candidates
    L = trues(N) # Tracks columns of AtA aloready computed


    # Homotopy loop
    it = 0
    while true
        # Compute intermediary values
        @views iAtA = pinv(AtA[K,K])
        a1[K] .= iAtA*Atb[K] # b* = (AtA)-1 Atb
        a2[K] .= iAtA*theta[K] # Δb = (AtA)-1 At*theta
        a3[.~K] .= Atb[.~K] .- AtA[.~K,K] * a1[K] # Ar* = A(b - Atb*)
        a4[.~K] .= sign.(a3[.~K]) .- AtA[.~K,K] * a2[K] # sign(Ar*) - Δr = sign(Ar*) - AtA*Δb

        # Compute querries that match criterions
        q1 .= K .& (abs.(a2) .> tol) .& (a1 .* theta .< zero(acc)) # Δb != 0 and sign(b*) != sign(theta)
        q2 .= .~K .& (abs.(a4) .> tol)                             # sign(Ar*) - Δr != 0


        # Evaluate first unfeasibility point
        v[q1] .= a1[q1] ./ a2[q1]
        if any(q1) 
            max1, k1 = findmax(v[q1])
        else 
            max1 = zero(acc)
        end

        # Evaluate first insufficiency point
        v[q2] .= a3[q2] ./ a4[q2]
        if any(q2)
            max2,k2 = findmax(v[q2])
        else 
            max2 = zero(acc)
        end


        # Choose lambda
        lambda = max(max1,max2)
        if (verbose); println("\tlambda = ",round(lambda,digits=3),"\t\tsol = ",round.(a1,digits=2)) end

        if (metric(A,b,a1) || lambda < tol || it >= maxIter)
            a1[abs.(a1) .< 1e-12] .= zero(acc) # remove smallest values (numerical noise)
            return a1
        end
    
        # Update active set accordingly 
        if lambda > zero(acc)
            if (max1 > max2)
                k1 = (1:N)[q1][k1]
                K[k1] = false
                a1[k1] = zero(acc)
            else
                k2 = (1:N)[q2][k2]
                K[k2] = true
                theta[k2] = sign(a3[k2])
                if L[k2] 
                    @views AtA[:,k2] .= A'A[:,k2]
                    L[k2] = false
                end
            end
        end
        
        it += 1 
    end
end



"""
    This function gives the optimal solution C to the problem
        min_C ||X - XC||_F + lambda ||C||_1
    s.t. ||C(:,j)||_O = k  for all j = {1,2,...,N}

    INPUT :
    X = Data matrix [::Matrix]
    metric = function used to evaluate quality of the solution 
        and stop homotopy if solution is optimal [::Function metric(A,b,x) -> ::Bool]
    
    OPTIONAL :
    tol = Threshold used to stop homotopy based on metric function results [::Real]
    maxIter = Maximum number of iterations of ADMM [::Integer]
    verbose = If true, print lambda and sol on each iteration [::Bool]
    acc = Computing accuracy (type of inner variables, defaut is Float32) [::Float subtype]

    OUTPUT :
    C = NxN coefficient matrix [::Matrix{acc}]
    --------------------------------------------------------------------------
Copyright @ Florian Dubois, 2024
"""
function homotopy_solver(X,metric;tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    # Method container
    D,N = size(X);
    x = Vector{acc}(undef,D);
    C = Matrix{acc}(undef,N,N);

    # Solving problem columnwise with homotopy
    @inbounds for j in 1:N
        if j > 1; X[:,j-1] .= x end # Restore previous column 
        x .= X[:,j]; X[:,j] .= 0.0; # Save and eliminate current column

        C[:,j] .= homotopy(X,x,metric,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc);
    end

    X[:,end] .= x

    return C
end



"""
    This function gives the solution C to the problem
        min_C ||Y - YC||_F + lambda ||C||_1
        s.t. ||C(:,j)||_O = k  for all j = {1,2,...,N}

    using the k-nearest neighbours method, solving N sub-problem
        min_C ||Ynn[:,i] - Ynn*c_i||_F + lambda ||c_i||_1
        s.t. ||c_i||_O = k  for all j = {1,2,...,N}

    Y = data matrix [D x N Matrix of Float]
    Ynn = sub matrix of Y made of K nearest neighbours [D x K Matrix of Float]
    k = targeted sparsity of each column of solution C [Integer]
    C = coefficient matrix [N x N Matrix of Float]
    c_i = i-th columns of C (representation of point x_i) [N x 1 Array of Float]
-------------------------------------------------------------------------
Copyright @ Florian Dubois, 2024
"""
function knn_homotopy_solver(X,metric,kNN;tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    # Method container
    N = size(X,2);
    C = zeros(acc,N,N);

    # floor to N-1 if the number of neighbours K exceeds the availability 
    if (kNN >= N); kNN = N-1 end

    # Compute Distance matrix
    Xdist = pairwise(Euclidean(),X,dims=2)

    # Solving problem columnwise
    for j in axes(C,2)
        # Select the K nearest neighbours + self (distance=0), then reject self
        ind = [x for x in find_n_min(Xdist[:,j],kNN+1) if x != j] #finNmin defined in utils.jl

        # Solving with homotopy
        C[ind,j] .= homotopy(X[:,ind], X[:,j],metric,tol=tol,acc=acc,maxIter=maxIter,verbose=verbose);
    end
    return C
end




"""
    Expected output : 
        lambda = 179.0          sol = Float32[0.0, 0.0, 0.0, 0.0, 0.0]
        lambda = 89.2159        sol = Float32[0.0, 0.0, 0.0, 0.0, 0.17]
        lambda = 69.3997        sol = Float32[0.0, 0.83, 0.0, 0.0, 0.03]
        lambda = 11.2313        sol = Float32[0.64, 0.18, 0.0, 0.0, 0.07]
        lambda = 3.8818         sol = Float32[0.37, 0.33, 0.4, 0.0, -0.04]
        lambda = 1.4385         sol = Float32[0.49, 0.24, 0.29, 0.0, 0.0]
        lambda = 0.0363         sol = Float32[0.37, 0.33, 0.4, 0.0, -0.04]

    Returned 5-elements Vector{Float32} :
        [0.37211943  0.3309536  0.39878464  0.0021190643  -0.03643179]
"""
function _test_homotopy()
    A = [
        1 2 3 4 5;
        0 1 0 1 0;
        2 2 2 2 2;
        -1 1 -1 -1 1;
        3 2 2 1 1;
        5 5 5 1 5;
        2 4 8 16 32;
        8 8 1 1 1
    ];

    b = vec([2 -1 4 0 2 5 4 6]);

    # solve
    homotopy(A,b, (A,b,x)-> false ;maxIter = 10,verbose=true);
end







"""
    This function gives, for each criterions among a set, the optimal solution x of the problem
        min_x ||Ax-b||_F + lambda ||x||_1
    that meet it

    --------------------------------------------------------------------------
    Input :
    --------------------------------------------------------------------------
    A = data matrix [D x N Matrix of Float]
    b = target solution [D x 1 Array of Float]
    threshold = threshold used to stop homotopy based on metric function results [Float]
    metric = function used to evaluate quality of the solution. Must return a BitArray m 
        where m[i] = true if criterion i is met [Function (A,b,x) -> M x 1 Array of Bool]
    C = container for solution [N x M Matrix of Float]

    acc must be Float !!!

    --------------------------------------------------------------------------
    Output :
    --------------------------------------------------------------------------
    solution = set of solutions that meet metric criterions [N x M Matrix of Float]
    
-------------------------------------------------------------------------
Copyright @ Nicolas Nadisic, 2022 & Florian Dubois, 2024
"""
function broadcasted_homotopy(A::AbstractArray{<:Real,2},
                              b::AbstractVector{<:Real},
                              metric::Function,
                              m::Integer;
                              tol::Float64 = 1e-12,
                              acc::Type = Float32,
                              maxIter::Integer = 1000,
                              verbose::Bool=false)

    # Debug tools
    # if !(tol >0); error("Negative tollerance ! Try @assert tol > 0") end
    # if !(acc <: Float); error("Accuracy must be a Float ! Try @assert acc <: Float) end
    # if !(maxIter > 0); error("Negative maxIter ! Try @assert maxIter > 0") end

    # Init 
    N = size(A, 2)
    lambda = zero(acc)

    if !(A isa AbstractMatrix{acc}); A = convert(Matrix{acc},A) end # If A do not match accuracy type, convert
    if !(b isa AbstractVector{acc}); b = convert(Vector{acc},b) end# If b do not match accuracy type, convert

    AtA = Matrix{acc}(undef,N,N)
    Atb = A'b
    C = Matrix{acc}(undef,N,m)

    # Pre-allocate Containers
    K = falses(N) # active set (initial support is empty)
    theta = Vector{acc}(undef,N) # gradient vector of l1-norm
    v = Vector{acc}(undef,N) # Container for lambda candidates
    a1, a2, a3, a4 = zeros(acc,N),Vector{acc}(undef,N),Vector{acc}(undef,N),Vector{acc}(undef,N) # Container for b*, Δb, Ar* and sign(Ar*) - Δr
    q1,q2 = BitArray(undef,N),BitArray(undef,N) # Mask for feature candidates
    L = trues(N) # Tracks columns of AtA aloready computed
    m1,m2 = BitArray(undef,m),falses(m) # Tracks which criterion has been met


    # Homotopy loop
    it = 0
    while true
        # Compute intermediary values
        @views iA = pinv(AtA[K,K])
        a1[K] .= iA*Atb[K] # b* = (AtA)-1 Atb
        a2[K] .= iA*theta[K] # Δb = (AtA)-1 At*theta
        a3[.~K] .= Atb[.~K] .- AtA[.~K,K] * a1[K] # Ar* = A(b - Atb*)
        a4[.~K] .= sign.(a3[.~K]) .- AtA[.~K,K] * a2[K] # sign(Ar*) - Δr = sign(Ar*) - AtA*Δb

        # Compute querries that match criterions
        q1 .= K .& (abs.(a2) .> tol) .& (a1 .* theta .< zero(acc)) # Δb != 0 and sign(b*) != sign(theta)
        q2 .= .~K .& (abs.(a4) .> tol)                             # sign(Ar*) - Δr != 0


        # Evaluate first unfeasibility point
        v[q1] .= a1[q1] ./ a2[q1]
        if any(q1) 
            max1, k1 = findmax(v[q1])
        else 
            max1 = zero(acc)
        end

        # Evaluate first insufficiency point
        v[q2] .= a3[q2] ./ a4[q2]
        if any(q2)
            max2,k2 = findmax(v[q2])
        else 
            max2 = zero(acc)
        end


        # Choose lambda
        lambda = max(max1,max2)
        if (verbose); println("\tlambda = ",round(lambda,digits=3),"\t\tsol = ",round.(a1,digits=2)) end

        # Evaluate solution
        m1 .= metric(A,b,a1)
        if any(m1)
            # Save solution in container
            a1[abs.(a1) .< 1e-12] .= zero(acc) # remove smallest values (numerical noise)
            for i in eachindex(m1)
                if m1[i] && !m2[i]; C[:,i] .= a1 end
            end

            # Update registry and exit if all criterions met
            m2 .|= m1
            if all(m2); return C end
        end

        # Exit if homotopy converged or exceeded maxIter 
        if (lambda < tol || it >= maxIter)
            # Save last computed solution in criterions not met yet
            a1[abs.(a1) .< 1e-12] .= zero(acc) # remove smallest values (numerical noise)
            for i in eachindex(m2)
                if !m2[i]; C[:,i] .= a1 end
            end
            return C
        end
    
        # Update active set accordingly 
        if lambda > zero(acc)
            if (max1 > max2)
                k1 = (1:N)[q1][k1]
                K[k1] = false
                a1[k1] = zero(acc)
            else
                k2 = (1:N)[q2][k2]
                K[k2] = true
                theta[k2] = sign(a3[k2])
                if L[k2] 
                    @views AtA[:,k2] .= A'A[:,k2]
                    L[k2] = false
                end
            end
        end
    
        it += 1 
    end
end


#--------------------------------------------------------------------------
# This function gives the optimal solution C to the problem
#       min_C ||Y - YC||_F + lambda ||C||_1
#  s.t. ||C(:,j)||_O = k  for all j = {1,2,...,N}
#
# Y = data matrix [D x N Matrix of Float]
# k = targeted sparsity of each column of solution C [Integer]
# C = coefficient matrix [N x N Matrix of Float]
#--------------------------------------------------------------------------
# Copyright @ Florian Dubois, 2024
#--------------------------------------------------------------------------
function broadcasted_homotopy_solver(X,metric,m;tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    # Method container
    D,N = size(X);
    x = Vector{acc}(undef,D)
    C = Array{acc,3}(undef,N,N,m)

    # Solving problem columnwise with homotopy
    @inbounds for j in 1:N
        if j > 1; X[:,j-1] .= x end # Restore previous column 
        x .= X[:,j]; X[:,j] .= 0.0; # Save and eliminate current column

        C[:,j,:] .= broadcasted_homotopy(X,x,metric,m,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc);
    end

    X[:,end] .= x

    return C
end



"""
    This function gives the solution C to the problem
        min_C ||Y - YC||_F + lambda ||C||_1
        s.t. ||C(:,j)||_O = k  for all j = {1,2,...,N}

    using the k-nearest neighbours method, solving N sub-problem
        min_C ||Ynn[:,i] - Ynn*c_i||_F + lambda ||c_i||_1
        s.t. ||c_i||_O = k  for all j = {1,2,...,N}

    Y = data matrix [D x N Matrix of Float]
    Ynn = sub matrix of Y made of K nearest neighbours [D x K Matrix of Float]
    k = targeted sparsity of each column of solution C [Integer]
    C = coefficient matrix [N x N Matrix of Float]
    c_i = i-th columns of C (representation of point x_i) [N x 1 Array of Float]
-------------------------------------------------------------------------
Copyright @ Florian Dubois, 2024
"""
function broadcasted_knn_homotopy_solver(X,metric,m,kNN;tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    # Method container
    N = size(X,2);
    C = zeros(acc,N,N,m);

    # floor to N-1 if the number of neighbours K exceeds the availability 
    if (kNN >= N); kNN = N-1 end

    # Compute Distance matrix
    Xdist = pairwise(Euclidean(),X,dims=2)

    # Solving problem columnwise
    for j in axes(C,2)
        # Select the K nearest neighbours + self (distance=0), then reject self
        ind = [x for x in find_n_min(Xdist[:,j],kNN+1) if x != j] #finNmin defined in utils.jl

        # Solving with homotopy
        C[ind,j,:] .= broadcasted_homotopy(X[:,ind], X[:,j],metric,m,tol=tol,acc=acc,maxIter=maxIter,verbose=verbose);
    end

    return C
end


"""
    Expected output : 
        lambda = 179.0          sol = Float32[0.0, 0.0, 0.0, 0.0, 0.0]
        lambda = 89.2159        sol = Float32[0.0, 0.0, 0.0, 0.0, 0.17]
        lambda = 69.3997        sol = Float32[0.0, 0.83, 0.0, 0.0, 0.03]
        lambda = 11.2313        sol = Float32[0.64, 0.18, 0.0, 0.0, 0.07]
        lambda = 3.8818         sol = Float32[0.37, 0.33, 0.4, 0.0, -0.04]
        lambda = 1.4385         sol = Float32[0.49, 0.24, 0.29, 0.0, 0.0]
        lambda = 0.0363         sol = Float32[0.37, 0.33, 0.4, 0.0, -0.04]

    Returned 5x6 Matrix{Float32} :
        [0.0       0.0        0.642592    0.371796   0.372119     0.372119  ]
        [0.0       0.830922   0.18456     0.331257   0.330954     0.330954  ]
        [0.0       0.0        0.0         0.398736   0.398785     0.398785  ]
        [0.0       0.0        0.0         0.0       -0.00211906  -0.00211906]
        [0.165587  0.0287659  0.0709978  -0.037498  -0.0364318   -0.0364318 ]
"""
function _test_br_homotopy()
    A = [
        1 2 3 4 5;
        0 1 0 1 0;
        2 2 2 2 2;
        -1 1 -1 -1 1;
        3 2 2 1 1;
        5 5 5 1 5;
        2 4 8 16 32;
        8 8 1 1 1
    ];

    b = [2,-1,4,0,2,5,4,6];
    thrs = (1,2,3,4,5,6)

    # solve
    C = broadcasted_homotopy(A,b, (A,b,x)->(sum(x .!= 0.0) .== thrs), length(thrs);maxIter = 10,verbose=true);

    display(C)
end