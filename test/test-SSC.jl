function test_ssc()
    N = 6 # size parameters

    # Progress bar
    i,ntest = 0,13;
    prefix = "\r\u1b[2K\t\t Testing SSC ... "
    i = progress(i,ntest,prefix)

    # Define tools
    refX = ones(Float32,3*N,2*N);
    refX[1:2*N,1:2*N] += I;
    X = copy(refX)


    # Test base
    @test ssc(X) isa Vector{Int64};                                 
    i = progress(i,ntest,prefix)

    # Test @view
    @test ssc(@view X[1:N,1:N]) isa Vector{Int64};                  
    i = progress(i,ntest,prefix)

    # Test spectral_clustering(nclust!=0)
    @test ssc(X,nclust=3) isa Vector{Int64};                             
    i = progress(i,ntest,prefix)

    # Test admm_lasso_affine()
    @test ssc(X,affine=true) isa Vector{Int64};                     
    i = progress(i,ntest,prefix)

    # Test admm_outliers_lin(), error_lin_sys()
    @test ssc(X,outliers=true) isa Vector{Int64};                   
    i = progress(i,ntest,prefix)

    # Test admm_outliers_affine()
    @test ssc(X,affine=true,outliers=true) isa Vector{Int64};       
    i = progress(i,ntest,prefix)

    # Test _thr(rho!=1)
    @test ssc(X,rho=0.6) isa Vector{Int64};                         
    i = progress(i,ntest,prefix)

    # Test data_projection(r!=0)
    @test ssc(X,r=30) isa Vector{Int64};                            
    i = progress(i,ntest,prefix)

    # Test admm_lasso_lin(alpha1=1e6)
    @test ssc(X,alpha=(1e6,20,20)) isa Vector{Int64};               
    i = progress(i,ntest,prefix)

    # Test admm_outliers_lin(alpha3=1e6)
    @test ssc(X,alpha=(10,20,1e6),outliers=true) isa Vector{Int64}; 
    i = progress(i,ntest,prefix)

    # Test change on trh
    @test ssc(X,thr=(6e-4,4e-3),outliers=true) isa Vector{Int64};   
    i = progress(i,ntest,prefix)

    # Test change on accuracy
    @test ssc(X,acc=Float64) isa Vector{Int64};                     
    i = progress(i,ntest,prefix,)

    # Test data integrity
    @test X == refX;   
    i = progress(i,ntest,prefix,"\n")
end