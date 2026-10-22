function test_misclassification()
    # Progress bar
    i,ntest = 0,5;
    prefix = "\r\u1b[2K   Testing misclassification ... ";
    i = progress(i,ntest,prefix);

    # Define tools
    grps = [1,1,2,2,3,3,4,4];
    s = [1,1,2,2,3,3,4,4];

    # Test Base
    @test misclassification(grps,s) isa Tuple{<:Real,Vector{<:Integer}};        
    i = progress(i,ntest,prefix)

    # Test misclassification(n!=0)
    @test misclassification(grps,s,nclust=4) isa Tuple{<:Real,Vector{<:Integer}};    
    i = progress(i,ntest,prefix)

    # Test excat clustering
    grps = [1,1,3,3,2,2,4,4];
    @test misclassification(grps,s) == (0.0,[1,3,2,4]);                         
    i = progress(i,ntest,prefix)

    # Test under clustering
    grps = [1,1,1,1,2,2,3,3];
    @test misclassification(grps,s) == (0.25,[1,3,4,2]);                        
    i = progress(i,ntest,prefix)

    # Test under clustering
    grps = [4,5,1,1,2,2,3,3];
    @test misclassification(grps,s) == (0.125,[2,3,4,1,5]);                     
    i = progress(i,ntest,prefix,"\n")
end