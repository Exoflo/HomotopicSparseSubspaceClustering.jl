# using SparseArrays: sparse

# include("homotopy.jl")
# include("metrics.jl")


"""
    This is the function to solve a subspace clustering problem using the HSSC algorithm
    It calls the homotopy sparse optimization program to find the solution matrix (C) 
    then calls the spectral clustering algorithm to get a clustering of the data (grps) 
    and finaly computes the clustering error (missrate) regarding the ground truth (s)
    
    Input :
    X = data matrix [D x N Matrix of Float]
    metric = function used to evaluate quality of 
       the solution and stop homotopy if solution is optimal [Function (A,b,x) -> Bool]
    
    nclust = Expected number of clusters [Integer]
    
    Output :
    [missrate] = clustering error (1 - accuracy) [Float in [0,1]]
    C = coefficient matrix obtained by HSSC [N x N Matrix of Float]
    [solve verify] = time (in second) to solve the problem and verify the solution [1 x 2 Array of Float]
--------------------------------------------------------------------------
    Copyright @ Florian Dubois, 2024
"""
function hssc(X::AbstractArray{<:Real,2},
              metric::Function;
              nclust::Integer = 0,
              tol::Real = 1e-12,
              maxIter::Integer = 1000,
              verbose::Bool = false,
              acc::Type = Float32)

    C = abs.(sparse(homotopy_solver(X,metric,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)))
    A = C + C'
    build_adjacency!(A)
    grps = spectral_clustering(A,nclust);

    return grps
end



#--------------------------------------------------------------------------
# This function is the equivalent of HSSC with metric = ksparse(k)
#--------------------------------------------------------------------------
function khssc(X,k;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return hssc(X,ksparse(k),nclust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



#--------------------------------------------------------------------------
# This function is the equivalent of HSSC with metric = relerror(e)
#--------------------------------------------------------------------------
function ehssc(X,e;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return hssc(X,relerror(e),nlust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



"""
    This is the function to solve a subspace clustering problem using the kNN-HSSC algorithm
    It first start by choosing the closest k nearest neighbours Xnn among data matrix X
    then calls the homotopy sparse optimization program to find the solution matrix (C) 
    then calls the spectral clustering algorithm to get a clustering of the data (grps) 
    and finaly computes the clustering error (missrate) regarding the ground truth (s)
    
    Input :
    X = data matrix [D x N Matrix of Float]
    s = clustering ground-truth vector [N x 1 Array of Integer]
    metric = function used to evaluate quality of 
       the solution and stop homotopy if solution is optimal [Function (A,b,x) -> Bool]
    N = number of nearest neighbours to consider [Integer]
    
    Output :
    [missrate] = clustering error (1 - accuracy) [Float in [0,1]]
    C = coefficient matrix obtained by HSSC [N x N Matrix of Float]
    [solve verify] = time (in second) to solve the problem and verify the solution [1 x 2 Array of Float]
--------------------------------------------------------------------------
    Copyright @ Florian Dubois, 2024
"""
function knn_hssc(X::AbstractArray{<:Real,2},
                  metric::Function,
                  kNN::Integer;
                  nclust::Integer = 0,
                  tol::Real = 1e-12,
                  maxIter::Integer = 1000,
                  verbose::Bool = false,
                  acc::Type = Float32)

    C = abs.(sparse(knn_homotopy_solver(X,metric,kNN,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)))
    A = C + C'
    build_adjacency!(A)
    grps = spectral_clustering(A,nclust);

    return grps
end



#--------------------------------------------------------------------------
# This function is the equivalent of knn_hssc() with metric = ksparse(k)
#--------------------------------------------------------------------------
function knn_khssc(X,k,kNN;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return knn_hssc(X,ksparse(k),kNN,nclust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



#--------------------------------------------------------------------------
# This function is the equivalent of knn_hssc() with metric = relerror(e)
#--------------------------------------------------------------------------
function knn_ehssc(X,e,kNN;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return knn_hssc(X,relerror(e),kNN,nlust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end






"""
    This is the function to solve a subspace clustering problem using a broadcasted version of the HSSC algorithm
    It calls the homotopy sparse optimization program to find the solution matrix (C) 
    then calls the spectral clustering algorithm to get a clustering of the data (grps) 
    and finaly computes the clustering error (missrate) regarding the ground truth (s)
    
    Input :
    X = data matrix [D x N Matrix of Float]
    s = clustering ground-truth vector [N x 1 Array of Integer]
    metric = function used to evaluate quality of 
       the curent solution and extract it if optimal regarding any given parameter [Function (A,b,x) -> Vector of Bool]
    m : number of threshold values to evaluate in the metric function
    
    Output :
    [missrate] = clustering errors (1 - accuracy) [1 x m Array of Float in [0,1]]
    C = coefficient matrixes obtained by HSSC [N x N x m Tensor of Float]
    [solve verify] = times (in second) to solve the problem and verify the solution [1 x 2 Array of Float]
--------------------------------------------------------------------------
    Copyright @ Florian Dubois, 2024
"""
function broadcasted_hssc(X::AbstractArray{<:Real,2},
                          metric::Function,
                          m::Integer;
                          nclust::Integer = 0,
                          tol::Real = 1e-12,
                          maxIter::Integer = 1000,
                          verbose::Bool = false,
                          acc::Type = Float32)

    Cs = broadcasted_homotopy_solver(X,metric,m,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
    C = [abs.(sparse(c)) for c in eachslice(Cs,dims=3)] # Vector of sparse matrixes with positive value ( C = [m x N x N]), where N is the number of columns of X
    
    A = C + transpose.(C)
    build_adjacency!.(A)

    grps = spectral_clustering.(A,nclust);

    return grps
end



#--------------------------------------------------------------------------
# This function is the equivalent of broadcasted_hssc() with metric = ksparse(thrs)
#--------------------------------------------------------------------------
function br_khssc(X,thrs,m;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return broadcasted_hssc(X,ksparse(thrs),m,nclust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



#--------------------------------------------------------------------------
# This function is the equivalent of broadcasted_hssc() with metric = relerror(thrs)
#--------------------------------------------------------------------------
function br_ehssc(X,thrs,m;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return broadcasted_hssc(X,relerror(thrs),m,nlust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



"""
    This is the function to solve a subspace clustering problem using the kNN-HSSC algorithm
    It first start by choosing the closest k nearest neighbours Xnn among data matrix X
    then calls the homotopy sparse optimization program to find the solution matrix (C) 
    then calls the spectral clustering algorithm to get a clustering of the data (grps) 
    and finaly computes the clustering error (missrate) regarding the ground truth (s)
    
    Input :
    X = data matrix [D x N Matrix of Float]
    s = clustering ground-truth vector [N x 1 Array of Integer]
    metric = function used to evaluate quality of 
       the solution and stop homotopy if solution is optimal [Function (A,b,x) -> Bool]
    m : number of threshold values to evaluate in the metric function
    N = number of nearest neighbours to consider [Integer]
    
    Output :
    [missrate] = clustering error (1 - accuracy) [Float in [0,1]]
    C = coefficient matrix obtained by HSSC [N x N Matrix of Float]
    [solve verify] = time (in second) to solve the problem and verify the solution [1 x 2 Array of Float]
--------------------------------------------------------------------------
    Copyright @ Florian Dubois, 2024
"""
function broadcasted_knn_hssc(X::AbstractArray{<:Real,2},
                              metric::Function,
                              m::Integer,
                              kNN::Integer;
                              nclust::Integer = 0,
                              tol::Real = 1e-12,
                              maxIter::Integer = 1000,
                              verbose::Bool = false,
                              acc::Type = Float32)

    Cs = broadcasted_knn_homotopy_solver(X,metric,m,kNN,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
    C = [abs.(sparse(c)) for c in eachslice(Cs,dims=3)] # Vector of sparse matrixes with positive value ( C = [m x N x N]), where N is the number of columns of X
    
    A = C + transpose.(C)
    build_adjacency!.(A)
    
    grps = spectral_clustering.(A,nclust);

    return grps
end



#--------------------------------------------------------------------------
# This function is the equivalent of broadcasted_knn_hssc() with metric = ksparse(thrs)
#--------------------------------------------------------------------------
function br_knn_khssc(X,thrs,m;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return broadcasted_knn_hssc(X,ksparse(thrs),m,nclust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end



#--------------------------------------------------------------------------
# This function is the equivalent of broadcasted_knn_hssc() with metric = relerror(thrs)
#--------------------------------------------------------------------------
function br_knn_ehssc(X,thrs,m,kNN;nclust=0,tol=1e-12,maxIter=1000,verbose=false,acc=Float32)
    return broadcasted_knn_hssc(X,relerror(thrs),m,kNN,nlust=nclust,tol=tol,maxIter=maxIter,verbose=verbose,acc=acc)
end