# using SparseArrays: sparse

# include("ADMM.jl)

"""
    This is the function to call the sparse optimization program, to call the spectral clustering algorithm and to compute the clustering error.
    
    input : 
    X : DxN Data matrix
    nclust : Expected number of clusters
    outlier : if true then use the outlier framework
    affine : if true then use the affine constraint
    rho : percentage of column to keep when thresholding the coefficient matrix C 
    r : projection dimension, if r = 0, then no projection
    alpha : scaling coeficient to compute mu2
    thr : stopping threshold for the coefficient error ||Z-C|| and the linear system error ||Y-YZ||
    maxIter : maximum number of iterations of ADMM
    verbose : if true then print error recap after solving
    acc : computing accuracy (type of inner variables, defaut is Float64)

    output :
    missrate : clustering error
    CMat : coefficient matrix
--------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function ssc(X::AbstractArray{<:Real,2};
             nclust::Integer = 0,
             outliers::Bool = false,
             affine::Bool = false,
             rho::Real = 1.0,
             r::Integer = 0,
             alpha::Tuple{<:Real,<:Real,<:Real} = (20,20,20),
             thr::Tuple{<:Real,<:Real} = (2e-4,2e-4),
             maxIter::Integer = 300,
             verbose::Bool = false,
             acc::Type = Float32)

    Xp = data_projection(X,r);
    C = sparse(admm(Xp,outliers=outliers,affine=affine,alpha=alpha,thr=thr,maxIter=maxIter,verbose=verbose,acc=acc));
    build_adjacency!(_thr(C,rho));
    C .+= C'

    return spectral_clustering(C,nclust)
end