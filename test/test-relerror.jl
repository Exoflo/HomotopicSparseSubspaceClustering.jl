function test_relerror()
    N = 10 # size parameters
    
    # Progress bar
    i,ntest = 0,12;
    prefix = "\r\u1b[2K\t    Testing relerror ... ";
    i = progress(i,ntest,prefix);

    # Define tools
    A = ones(2*N,N);
    b = ones(2*N);
    x = ones(N);
    xz = zeros(N);

    m = relerror(0.1);
    br_m = relerror([0.1,0.05,0.01]);

    # Test relerror integrity
    @test m isa Function                                    
    i = progress(i,ntest,prefix)

    # Test relerror broadcasted
    @test br_m isa Function                                 
    i = progress(i,ntest,prefix)

    # Test metric return type
    @test m(A,b,x) isa Bool                                 
    i = progress(i,ntest,prefix)

    # Test broadcasted metric return type
    @test br_m(A,b,x) isa BitVector                         
    i = progress(i,ntest,prefix)

    # Test relerror(0.1) on the null vector
    @test m(A,b,xz) == false                                
    i = progress(i,ntest,prefix)
    
    # Test relerror([0.1,0.05,0.01]) on the null vector
    @test br_m(A,b,xz) == [false,false,false]               
    i = progress(i,ntest,prefix)

    # Test relerror(0.1) on a 90% reconstructed vector
    xz[1] = 0.9
    @test m(A,b,xz) == true                                 
    i = progress(i,ntest,prefix)

    # Test relerror([0.1,0.05,0.01]) on a 90% reconstructed vector
    @test br_m(A,b,xz) == [true,false,false]                
    i = progress(i,ntest,prefix)

    # Test relerror(0.1) on a 95% reconstructed vector
    xz[2] = 0.05
    @test m(A,b,xz) == true                                 
    i = progress(i,ntest,prefix)

    # Test relerror([0.1,0.05,0.01]) on a 95% reconstructed vector
    @test br_m(A,b,xz) == [true,true,false]                 
    i = progress(i,ntest,prefix)

    # Test relerror(0.1) on a 99% reconstructed vector
    xz[3] = 0.04
    @test m(A,b,xz) == true                                 
    i = progress(i,ntest,prefix)

    # Test relerror([0.1,0.05,0.01]) on a 99% reconstructed vector
    @test br_m(A,b,xz) == [true,true,true]                  
    i = progress(i,ntest,prefix,"\n")
end