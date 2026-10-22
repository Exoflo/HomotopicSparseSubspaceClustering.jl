
using Arpack: eigs                              # Usage : spectral_clustering()
using Hungarian: munkres                        # Usage : misclassification()
using LinearAlgebra: I, Symmetric, normalize!   # Usage : error_lin_sys(), spectral_clustering()
using ParallelKMeans: kmeans, Hamerly           # Usage : spectral_clustering()
using SparseArrays: SparseMatrixCSC, dropzeros  # Usage : build_adjacency!(), spectral_clustering(), _thr()
using TSVD: tsvd                                # Usage : data_projection()




"""
    This function takes a NxN coefficient matrix and returns a NxN adjacency
    matrix by choosing the K strongest connections in the similarity graph

    C : NxN coefficient matrix 
    K : Number of strongest edges to keep. If K = 0, keep all edges 

    Usage : SSC.jl - SCC
-------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function build_adjacency!(C::AbstractArray{<:Real,2},K::Integer = 0)
    C .= abs.(C);
    if K == 0 || K > size(C,1)
        foreach(col -> col ./= maximum(col),eachcol(C));
    else
        for col in eachcol(C)
            Ind = find_n_max(col,K);
            max = maximum(@view col[Ind])
            col[Ind] ./= max
        end
    end
end

# Redefinition for sparse matrices
function build_adjacency!(C::SparseMatrixCSC,K::Integer = 0)
    colind = C.colptr
    C.nzval .= abs.(C.nzval)
    if K == 0
        @inbounds for i in 1:C.n
            if colind[i] < colind[i+1]
                col = @view C.nzval[colind[i]:colind[i+1]-1];
                col ./= maximum(col);
            end
        end
    else
        @inbounds for i in 1:C.n
            if colind[i] < colind[i+1]
                col = @view C.nzval[colind[i]:colind[i+1]-1];
                Ind = find_n_max(col,K);
                max = maximum(@view col[Ind]);
                col[Ind] ./= max;
            end
        end
    end
end




"""
    This function takes a DxN matrix of N data points in a D-dimensional 
    space and returns the regularization constant of the L1 norm

    lambda = min_i max_(j != i) |y_i * y_j|

    Y: DxN data matrix
    lambda: regularization parameter for lambda*||C||_1 + 0.5 ||Y-YC||_F^2

    Usage : ADMM.jl - admm_lasso_lin - admm_lasso_affine - admm_outliers_lin - admm_outliers_affine
-------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function _compute_lambda(Y::AbstractArray{T}) where{T}
    mini,maxi = typemax(T),zero(T)
    D = size(Y,1)
    for idx in eachindex(Y)
        i = idx % D
        j = div(idx,D) + 1

        y = ifelse(i == j,0.0, abs(Y[idx])) # Skip C[i,i] 

        if (y > maxi); maxi = y end
        if (i == 0); mini = min(mini,maxi); maxi = zero(T) end # Update mini and reset maxi
    end
    return mini
end




"""
    This function takes the D x N data matrix with columns indicating
    different data points and project the D dimensional data into a r
    dimensional subspace using PCA

    X: D x N matrix of N data points
    r: dimension of the PCA projection, if r = 0, then no projection
    Xp: r x N matrix of N projectred data points

    Usage : SSC.jl - SCC
-------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function data_projection(X::AbstractArray{<:Real,2},r::Integer=0) 
    if (r <= 0) return X end
    U,_,_ = tsvd(X,r)
    return U' * X;
end



"""
    This function computes the maximum error between elements of two 
    coefficient matrices

    C: NxN coefficient matrix
    Z: NxN coefficient matrix
    err: infinite norm error between vectorized C and Z

    Usage : ADMM.jl - admmLasso_mat_func - ADMM.jl - admmOutlier_mat_func
-------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function error_coef(Z::AbstractArray,C::AbstractArray)
    max_err = 0.0
    for i in eachindex(Z)
        x = abs(Z[i] - C[i])
        if max_err < x; max_err = x end
    end
    return max_err
end


"""
    This function computes the maximum L2-norm error among the columns of the residual of a linear system 

    Y: DxN data matrix of N data point in a D-dimensional space
    Z: NxN sparse coefficient matrix
    err: maximum L2-norm of the columns of Y-YZ 

    Usage : ADMM.jl - admmLasso_mat_func - ADMM.jl - admmOutlier_mat_func
-------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function error_lin_sys(Y::AbstractArray{<:Real,2},Z::AbstractArray{<:Real,2})    # Y = [Y1[NxN] Y2[NxK]] = [NxD]
    D,N = size(Z);                                                               # Z = [Z1[NxN];Z2[KxN]] = [DxN]
    if D > N
        P1 = @views Y[:,N+1:end]*Z[N+1:end,:];                                   # P1 = Y1 - Y2*Z2
        P1 .-= @view Y[:,1:N]
        P2 = @views Y[:,1:N]*Z[1:N,:];                                           # P2 = Y1*Z1 - P1 = Y1*Z1 - Y1 + Y2*Z2 = Y1(Z1-I) + Y2*Z2
        P2 .+= P1
        return maximum(norm.(eachcol(P2)) ./ norm.(eachcol(P1)));                # P2 / P1
    else
        return maximum(norm.(eachcol(Y*(Z-I))) ./ norm.(eachcol(Y)));
    end
end




"""
    This function takes the groups resulted from spectral clutsering and the
    ground truth to compute the minimal misclassification rate.
    Number of classes don't need to be the same
    
    grps : Nx1 original classification (Vector{Integer})
    s : Nx1 ground truth vector (Vector{Integer})
    nclust : Maximal number of classes in grps or s (Integer)

    Missrate: misclassification rates (Float)
    Perm : Permutation to apply on grps labels to have minimal Missrate regarding s [Vector 1xn]
            Perm[i] = j means that changing label i to label j in grps gives the maximum class overlap
    Usage : SSC.jl - SCC
--------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function misclassification(grps::AbstractArray{<:Integer},
                            s::AbstractArray{<:Integer};
                            nclust::Integer = 0)

    if (nclust==0); # Compute extremas if n not given
        ming,maxg = extrema(grps)
        mins,maxs = extrema(s)
        nclust = max(maxg,maxs)
        if (ming < 1); error("Warning : classification labels must be integer in {1...n} ! 0 is not a valid class label, check 'grps' variable") end 
        if (mins < 1); error("Warning : classification labels must be integer in {1...n} ! 0 is not a valid class label, check 's' variable")    end
    end

    AccM = zeros(Int32,nclust,nclust) # AccM[prediction,ground truth]
    for i in eachindex(s)
        AccM[grps[i],s[i]] -= 1
    end
    Assign = (munkres(AccM) .== STAR)  # Solution to the Hungarian problem
    good = -sum(AccM .* Assign)                  # Number of correctly clustered points
    return 1 - good/length(s),Assign*(1:nclust);      # Return Missrate and Perm
end




"""
    This function takes an adjacency matrix of a graph and computes the 
    clustering of the nodes using the spectral clustering algorithm of 
    Ng, Jordan and Weiss.

    CMat: NxN adjacency matrix
    n: number of groups for clustering
    groups: N-dimensional vector containing the memberships of the N points 
    to the n groups obtained by spectral clustering

    Usage : SSC.jl - SCC
--------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function spectral_clustering(CKSym::SparseMatrixCSC, n::Integer = 0)
    # Normalized spectral clustering according to Ng & Jordan & Weiss  https://arxiv.org/pdf/0711.0189
    # using Normalized Symmetric Laplacian L = I - D^{-1/2} W D^{-1/2}
    DN = inv.(sqrt.(sum(CKSym,dims=1)) .+ eps());
    L = Symmetric(DN' .* CKSym .* DN - I);

    # If no prior on number of clusters, deduce from eigenvalues
    if n < 1 
        val::Vector{Float64},vect::Matrix{Float64},_,_,_,_ = eigs(L,nev=size(CKSym,1)-1,sigma=eps(),tol=1.0e-12,check=2)

        while true  # Find the number of relevent eigenvalues
            n+=1
            if (abs(val[n]) > 0.01) break end  # Hardcoded value of 0.01 !!!
        end
        if (n > 1); n -= 1 end

        kerN = (@view vect[:,1:n])' .+ eps()
    else
        kerN::Matrix{Float64} = eigs(L,nev=n,sigma=eps(),tol=1.0e-12,check=2)[2]' .+ eps();
    end

    foreach(normalize!,eachcol(kerN))
    groups = kmeans(Hamerly(),kerN,n,max_iters=1000).assignments; # display=:iter
    return groups
end



"""
    This function keeps the largest entries of each columns of C such that sum(col(C')) >= rho*sum(col(C))
    If rho >= 1, return C unchanged

    C : Matrix to threshold
    rho : percentage of sum(col) to keep
-----------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
Usage : SSC.jl - SCC
"""
function _thr(C::SparseMatrixCSC{T,I},rho = 1) where{T,I}
    if (rho >= 1 || rho < 0) return C end

    @inbounds for i in axes(C,2)
        col = abs.(C[:,i])
        sumcol = sum(col)
        # if column empty, skip
        if (sumcol == 0); continue end
        # if not, only keep the top rho% of the sum of the coefficients 
        obj = rho*sumcol
        ind = sortperm((col.nzval))
        ii = 1
        # Remove smallest until objective is reached
        while sumcol - col.nzval[ind[ii]] > obj
            C[col.nzind[ind[ii]],i] = zero(T)
            sumcol -= col.nzval[ind[ii]] 
            ii+=1
        end
    end

    return dropzeros(C)
end





"""
    thr_rect(x::T,thr::T) -> rectx::T where{T<:Number}

    This function return a rectified linear image of x regarding threshold thr (>=0)
                   if   x  <= -thr                        -> return x - (-thr)  
                   if         -thr <=  x  <=  thr         -> return zero(T)
                   if                         thr <=  x   -> return x - (+thr)

                            PLOT thr_rect(x,thr): 

                                    |         /
                                    |        /
                           -thr     |       /
           - - - - - - - - - +------0------+ - - - - - - - - -> x
                            /       |     +thr
                           /        |
                          /         |

    x: number to threshold (<:Number)
    thr : threshold (<:Number)

    Usage : ADMM.jl - admmLassoLin_mat_func,admmLassoAffine_mat_func,admmOutlierLin_mat_func,admmOutlierAffine_mat_func
--------------------------------------------------------------------------
Copyright @ Florian Dubois, 2026
"""
function thr_rect(x,thr)
    if x <= thr
        if x < -thr  
            return x + thr  # if x < -1/mu2 
        else            
            return zero(x)  # if -1/mu2 <= x <= 1/mu2
        end
    else
        return x - thr      # if x > 1/mu2 
    end
end




"""
    findNmin(A,n) -> ind::Array{CartesianIndex}

    Find the `n` minimal value in array `A` and return the list `ind` of their indexes
    If n >= length(A), return sortperm(A) (ind[1] give the index of the minimal element and ind[end] the index of the maximal one)

    Usage : HSSC.jl - kNN_HSCC
-----------------------------------------------------------------------
Copyright @ JM_Beckers, https://discourse.julialang.org/t/find-n-smallest-values-in-an-n-dims-array/81092/19
"""
function find_n_min(V::AbstractVector, n::Integer)
    # should someone ask more elements than array size, just sort array
    if n >= length(V)
        return 1:length(V)
    end

    # otherwise 
    ind=collect(1:n)
    mymax,getout=findmax(@view V[ind])

    @inbounds for j in (n+1):length(V)
        if V[j] < mymax
            ind[getout]=j
            mymax,getout=findmax(@view V[ind])
        end
    end

    return ind
end



"""
    findNmax(V,n) -> ind::Array{CartesianIndex}

    Find the `n` maximal value in array `V` and return the list `ind` of their indexes
    If n >= length(V), return 1:length(V) 

    Usage : utils.jl - BuildAdjacencyMatrix
-----------------------------------------------------------------------
Copyright @ Florian Dubois, inspired by JM_Beckers, https://discourse.julialang.org/t/find-n-smallest-values-in-an-n-dims-array/81092/13
"""
function find_n_max(V::AbstractVector, n::Integer)
    # should someone ask more elements than array size, just sort array
    if n >= length(V)
        return 1:length(V)
    end

    # otherwise 
    ind=collect(1:n)
    mymin,getout=findmin(@view V[ind])

    @inbounds for j in (n+1):length(V)
        if V[j] > mymin
            ind[getout]=j
            mymin,getout=findmin(@view V[ind])
        end
    end
    
    return ind
end