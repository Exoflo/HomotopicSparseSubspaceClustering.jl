function test_hssc()
    N = 10 # size parameters
    
    # Progress bar
    i,ntest = 0,7;
    prefix = "\r\u1b[2K\t\tTesting HSSC ... ";
    i = progress(i,ntest,prefix);

    # Define tools
    refX = ones(Float32,3*N,2*N);
    refX[1:2*N,1:2*N] += I;
    X = copy(refX);
    k = 3;
    thr = [2,3,4];
    m = ksparse(k);
    yes = (A,b,x) -> true;
    no = (A,b,x) -> false;


    ################## HSSC ##################
    # Test base
    @test hssc(X,m) isa Vector{Int64};                      
    i = progress(i,ntest,prefix)

    # Test @view
    @test hssc((@view X[1:N,1:N]),m) isa Vector{Int64};     
    i = progress(i,ntest,prefix)

    # Test spectral_clustering(nclust!=0)
    @test hssc(X,m,nclust=3) isa Vector{Int64};                  
    i = progress(i,ntest,prefix)

    # Test metric always return true
    @test hssc(X,yes) isa Vector{Int64};                    
    i = progress(i,ntest,prefix)

    # Test metric always return false
    @test hssc(X,no) isa Vector{Int64};                     
    i = progress(i,ntest,prefix)

    # Test change on accuracy
    @test hssc(X,m,acc=Float64) isa Vector{Int64};          
    i = progress(i,ntest,prefix)

    # Test data integrity
    @test X == refX; @test m == ksparse(k);
    i = progress(i,ntest,prefix,"\n")




    ################## kNN_HSSC ##################
    # Edit Tools
    kNN = 20; i,ntest = 0,6;
    prefix = "\r\u1b[2K\t    Testing kNN_HSSC ... ";
    i = progress(i,ntest,prefix)

    # Test base
    @test knn_hssc(X,m,kNN) isa Vector{Int64};                  
    i = progress(i,ntest,prefix)

    # Test spectral_clustering(nclust!=0)
    @test knn_hssc(X,m,kNN,nclust=3) isa Vector{Int64};              
    i = progress(i,ntest,prefix)

    # Test metric always return true
    @test knn_hssc(X,yes,kNN) isa Vector{Int64};                
    i = progress(i,ntest,prefix)

    # Test metric always return false
    @test knn_hssc(X,no,kNN) isa Vector{Int64};                 
    i = progress(i,ntest,prefix)

    # Test change on accuracy
    @test knn_hssc(X,m,kNN,acc=Float64) isa Vector{Int64};      
    i = progress(i,ntest,prefix)

    # Test data integrity
    @test X == refX; @test m == ksparse(k);
    i = progress(i,ntest,prefix,"\n")



    ################## br_HSSC ##################
    # Edit Tools
    i = 0;
    prefix = "\r\u1b[2K    Testing broadcasted_HSSC ... ";
    i = progress(i,ntest,prefix)
    
    m = ksparse(thr); 
    br_yes = (A,b,x) -> [true,true,true]; br_no = (A,b,x) -> [false,false,false];
    

    # Test base
    @test broadcasted_hssc(X,m,3) isa Vector{Vector{Int64}};                
    i = progress(i,ntest,prefix)

    # Test spectral_clustering(nclust!=0)
    @test broadcasted_hssc(X,m,3,nclust=3) isa Vector{Vector{Int64}};            
    i = progress(i,ntest,prefix)

    # Test metric always return true
    @test broadcasted_hssc(X,br_yes,3) isa Vector{Vector{Int64}};           
    i = progress(i,ntest,prefix)

    # Test metric always return false
    @test broadcasted_hssc(X,br_no,3) isa Vector{Vector{Int64}};            
    i = progress(i,ntest,prefix)

    # Test change on accuracy
    @test broadcasted_hssc(X,m,3,acc=Float64) isa Vector{Vector{Int64}};    
    i = progress(i,ntest,prefix)

    # Test data integrity
    @test X == refX; @test m == ksparse(thr);
    i = progress(i,ntest,prefix,"\n")



    ################## br_kNN_HSSC ##################
    # Edit Tools
    i = 0;
    prefix = "\r\u1b[2KTesting broadcasted_kNN_HSSC ... ";
    i = progress(i,ntest,prefix)

    # Test base
    @test broadcasted_knn_hssc(X,m,3,kNN) isa Vector{Vector{Int64}};                
    i = progress(i,ntest,prefix)

    # Test spectral_clustering(nclust!=0)
    @test broadcasted_knn_hssc(X,m,3,kNN,nclust=3) isa Vector{Vector{Int64}};            
    i = progress(i,ntest,prefix)

    # Test metric always return true
    @test broadcasted_knn_hssc(X,br_yes,3,kNN) isa Vector{Vector{Int64}};           
    i = progress(i,ntest,prefix)

    # Test metric always return false
    @test broadcasted_knn_hssc(X,br_no,3,kNN) isa Vector{Vector{Int64}};            
    i = progress(i,ntest,prefix)

    # Test change on accuracy
    @test broadcasted_knn_hssc(X,m,3,kNN,acc=Float64) isa Vector{Vector{Int64}};    
    i = progress(i,ntest,prefix)

    # Test data integrity
    @test X == refX; @test m == ksparse(thr);
    i = progress(i,ntest,prefix,"\n")
end