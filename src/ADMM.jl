# include("utils.jl")

# using LinearAlgebra: diagind
# using Distances: opnorm


"""
    admm(X;<keyword arguments>)::Matrix{acc}

Return a NxN coefficient matrix `C` of the sparse representation of each column of `X` in terms of the rest of the points.

# Arguments
- `X::AbstractArray{<:Real,2}`: DxN data matrix
- `outliers::Bool = false`: If true then use the outlier framework
- `affine::Bool = false`: If true then use the affine constraint
- `alpha::Tuple{<:Real,<:Real,<:Real} = (20,20,20)`: Meta parameter > 1 that act as a scaling coeficient to compute mu1,mu2 and gamma
- `thr` = Stopping threshold for the coefficient error ||Z-C|| and the linear system error ||Y-YZ||
- `maxIter::Integer = 300`: Maximum number of iterations of ADMM
- `verbose::Bool = false`: If true, print errors recap after solving
- `acc::Type = Float32`: Computing accuracy (type of inner variables, defaut is Float32)

--------------------------------------------------------------------------
Copyright @ Ehsan Elhamifar, 2012
"""
function admm(X::AbstractArray{<:Real,2};
             outliers::Bool = false,
             affine::Bool = false,
             alpha::Tuple{<:Real,<:Real,<:Real} = (20,20,20),
             thr::Tuple{<:Real,<:Real} = (2e-4,2e-4),
             maxIter::Integer = 300,
             verbose::Bool = false,
             acc::Type = Float32)

    if outliers
        if affine
            return admm_outliers_affine(X,alpha,thr,maxIter,verbose,acc)
        else
            return admm_outliers_lin(X,alpha,thr,maxIter,verbose,acc)
        end
    else
        if affine
            return admm_lasso_affine(X,alpha,thr,maxIter,verbose,acc)
        else
            return admm_lasso_lin(X,alpha,thr,maxIter,verbose,acc)
        end
    end
end




function admm_lasso_lin(Y::AbstractArray{<:Real,2},
                        alpha::Tuple{<:Real,<:Real,<:Real},
                        thr::Tuple{<:Real,<:Real},
                        maxIter::Integer,
                        verbose::Bool,
                        T::Type)

    if !(Y isa Matrix{T}); Y .= convert(Matrix{T},Y) end
    alpha1,alpha2,_ = convert(Tuple{T,T,T},alpha);
    thr1,_ = thr;

    # setting penalty parameters for the ADMM
    N = size(Y,2);
    YtY = Y'Y;
    mu1 = alpha1/_compute_lambda(YtY);
    mu2 = alpha2;
    invmu2 = inv(mu2);

    # initialization
    C = zeros(T,N,N);
    Z = Matrix{T}(undef,N,N);
    Lambda2 = zeros(T,N,N);
    err1 = 10*thr1;
    A = inv(mu1.*YtY + mu2.*I);

    # ADMM iterations
    iter = 0;
    while ( err1 > thr1 && iter < maxIter )
        # updating Z
        Z .= mu2 .* A*(C-I .- Lambda2); # + I (useless due to next opperation)
        foreach(i -> Z[i] = zero(T),diagind(Z));
        # updating C
        C .= thr_rect.(Z .+ Lambda2,invmu2);
        # updating Lagrange multipliers
        Lambda2 .+= (Z .- C);
        # computing errors
        err1 = error_coef(Z,C);
        #
        iter += 1;
    end
    if verbose
        println("err1: ",round(err1,digits=4),", err2: ",round(error_lin_sys(Y,Z),digits=4),", iter: $iter");
    end
    return C
end



function admm_lasso_affine(Y::AbstractArray{<:Real,2},
                            alpha::Tuple{<:Real,<:Real,<:Real},
                            thr::Tuple{<:Real,<:Real},
                            maxIter::Integer,
                            verbose::Bool,
                            T::Type)

    if !(Y isa Matrix{T}); Y .= convert(Matrix{T},Y) end
    alpha1,alpha2,_ = convert(Tuple{T,T,T},alpha);
    thr1,_ = thr;
    N = size(Y,2);

    # setting penalty parameters for the ADMM
    YtY = Y'Y
    mu1 = alpha1/_compute_lambda(YtY);
    mu2 = alpha2;
    invmu2 = inv(mu2);

    # initialization
    C = zeros(T,N,N);
    Z = Matrix{T}(undef,N,N);
    Lambda2 = zeros(T,N,N);
    Lambda3 = zeros(T,1,N);
    err1 = err3 = 10*thr1;
    A = inv(mu1.*YtY + mu2.*I .+ mu2);
    
    # ADMM iterations
    iter = 0;
    while ( (err1 > thr1 || err3 > thr1) && iter < maxIter )
        # updating Z
        Z .= mu2 .* A*(C-I .- Lambda2 .- Lambda3); # + I (useless due to next opperation)
        foreach(i -> Z[i] = zero(T),diagind(Z));
        # updating C
        C .= thr_rect.(Z .+ Lambda2,invmu2);
        # updating Lagrange multipliers
        Lambda2 .+= (Z .- C);
        Lambda3 .+= sum(Z,dims=1) .- 1;
        # computing errors
        err1 = error_coef(Z,C);
        err3 = error_coef(sum(Z,dims=1),ones(1,N));
        #
        iter += 1;
    end
    if verbose 
        println("err1: ",round(err1,digits=4),", err2: ",round(error_lin_sys(Y,Z),digits=4),", err3: ",round(err3,digits=4),", iter: $iter");
    end
    return C
end



function admm_outliers_lin(Y::AbstractArray{<:Real,2},
                            alpha::Tuple{<:Real,<:Real,<:Real},
                            thr::Tuple{<:Real,<:Real},
                            maxIter::Integer,
                            verbose::Bool,
                            T::Type)

    if !(Y isa Matrix{T}); Y .= convert(Matrix{T},Y) end
    alpha1,alpha2,alpha3 = convert(Tuple{T,T,T},alpha);
    thr1,thr2 = thr;

    # precomputing coefficients
    D,N = size(Y);
    YtY = Y'Y;
    invgamma = opnorm(Y,1)/alpha3;
    mu1 = alpha1/_compute_lambda([YtY;invgamma*Y]);   # P = [Y invgamma*I] / P'Y = [Y'Y; invgamma*Y]
    mu2 = alpha2;
    invmu2 = inv(mu2);
    mug = mu1*invgamma;
    invD = inv(mug*invgamma + mu2);

    # initialization A
    A11 = inv((mu1 - mug^2*invD) .* YtY + mu2.*I);
    A21 = (-mug*invD) .* Y * A11;
    A12 = A21';
    A22 = (mug*invD)^2 .* Y * A11 * Y'+ invD .* I;

    # initialization other
    Zn,S1 = Matrix{T}(undef,N,N),Matrix{T}(undef,N,N)
    Cn,Lambda2n = zeros(T,N,N),zeros(T,N,N)
    Zd,S2,P2,ws = Matrix{T}(undef,D,N),Matrix{T}(undef,D,N),Matrix{T}(undef,D,N),Matrix{T}(undef,D,N)
    Cd,Lambda2d,Lambda1 = zeros(T,D,N),zeros(T,D,N),zeros(T,D,N)
    err1 = 10*thr1; err2 = 10*thr2;
    iter = 0;
    # ADMM iterations
    while ( (err1 > thr1 || err2 > thr2) && iter < maxIter )
        # updating Z
        S1 .= mu1 .* (YtY .+ Y'Lambda1) .+ mu2.*(Cn .- Lambda2n);
        S2 .= (mu1 * invgamma) .* (Y .+ Lambda1) + mu2 .* (Cd .- Lambda2d);
        Zn .= A11*S1 .+ A12*S2;
        Zd .= A21*S1 .+ A22*S2;
        foreach(i -> Zn[i] = zero(T),diagind(Zn));
        # updating C
        Cn .= thr_rect.(Zn .+ Lambda2n,invmu2);
        Cd .= thr_rect.(Zd .+ Lambda2d,invmu2);
        # updating Lagrange multipliers
        Lambda1 .+= Y .- Y*Zn .- invgamma.*Zd;
        Lambda2n .+= Zn .- Cn;
        Lambda2d .+= Zd .- Cd;
        # computing errors
        err1 = max( error_coef(Zn,Cn),error_coef(Zd,Cd) );
        P2 .= invgamma.*Zd .- Y; 
        ws .= Y*Zn .- P2;
        err2 = maximum(norm.(eachcol(ws)) ./ norm.(eachcol(P2)));
        #
        iter += 1;
    end
    if verbose
        println("err1: ",round(err1,digits=4),", err2: ",round(err2,digits=4),", iter: $iter");
    end

    return Cn
end



function admm_outliers_affine(Y::AbstractArray{<:Real,2},
                               alpha::Tuple{<:Real,<:Real,<:Real},
                               thr::Tuple{<:Real,<:Real},
                               maxIter::Integer,
                               verbose::Bool,
                               T::Type)

    if !(Y isa Matrix{T}); Y .= convert(Matrix{T},Y) end
    alpha1,alpha2,alpha3 = convert(Tuple{T,T,T},alpha);
    thr1,thr2 = thr;

    # precomputing coefficients
    D,N = size(Y);
    YtY = Y'Y;
    invgamma = opnorm(Y,1)/alpha3;
    mu1 = alpha1/_compute_lambda([YtY;invgamma*Y]);   # P = [Y invgamma*I] / P'Y = [Y'Y; invgamma*Y]
    mu2 = alpha2;
    invmu2 = inv(mu2);
    mug = mu1*invgamma;
    invD = inv(mug*invgamma + mu2);

    # initialization A
    A11 = inv((mu1 - mug^2*invD) .* YtY + mu2.*I .+ mu2);
    A12 = -A11*((mug*invD) .* Y');
    A21 = A12';
    A22 = (mug*invD)^2 .* Y * A11 * Y'+ invD .* I;

    # initialization other
    Zn,S1 = Matrix{T}(undef,N,N),Matrix{T}(undef,N,N)
    Cn,Lambda2n = zeros(T,N,N),zeros(T,N,N)
    Zd,S2 = Matrix{T}(undef,D,N),Matrix{T}(undef,D,N)
    Cd,Lambda2d,Lambda1 = zeros(T,D,N),zeros(T,D,N),zeros(T,D,N);
    Lambda3 = zeros(T,1,N);
    err1 = err3 = 10*thr1; err2 = 10*thr2;
    iter = 0;
    # ADMM iterations
    while ( (err1 > thr1 || err2 > thr2 || err3 > thr1) && iter < maxIter )
        # updating Z
        S1 .= mu1 .* (YtY .+ Y'Lambda1) .+ mu2.*(Cn .- Lambda2n .- Lambda3 .+ 1);
        S2 .= (mu1 .* invgamma) .* (Y .+ Lambda1) + mu2 .* (Cd .- Lambda2d);
        Zn .= A11*S1 .+ A12*S2;
        Zd .= A21*S1 .+ A22*S2;
        foreach(i -> Zn[i] = zero(T),diagind(Zn));
        # updating C
        Cn .= thr_rect.(Zn .+ Lambda2n,invmu2);
        Cd .= thr_rect.(Zd .+ Lambda2d,invmu2);
        # updating Lagrange multipliers
        Lambda1 .+= Y .- Y*Zn .- invgamma.*Zd;
        Lambda2n .+= Zn .- Cn;
        Lambda2d .+= Zd .- Cd;
        Lambda3 .+= sum(Zn,dims=1) .- 1;
        # computing errors
        err1 = max( error_coef(Zn,Cn),error_coef(Zd,Cd) );
        S2 .= invgamma.*Zd; S2 .-= Y; 
        ws = Y*Zn; ws  .-= S2;
        err2 = maximum(norm.(eachcol(ws)) ./ norm.(eachcol(S2)));
        err3 = error_coef(sum(Zn,dims=1),ones(T,N));
        #
        iter += 1;
    end
    if verbose
        println("err1: ",round(err1,digits=4),", err2: ",round(err2,digits=4),", err3: ",round(err3,digits=4),", iter: $iter");
    end

    return Cn
end