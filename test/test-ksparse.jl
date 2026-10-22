function test_ksparse()
    N = 10 # size parameters
    
    # Progress bar
    i,ntest = 0,12;
    prefix = "\r\u1b[2K\t     Testing ksparse ... ";
    i = progress(i,ntest,prefix);

    # Define tools
    A = ones(2*N,N);
    b = ones(2*N);
    x = ones(N);
    xz = zeros(N);

    m = ksparse(1);
    br_m = ksparse([1,2,3]);

    # Test ksparse integrity
    @test m isa Function                                    
    i = progress(i,ntest,prefix)

    # Test ksparse broadcasted
    @test br_m isa Function                                 
    i = progress(i,ntest,prefix)

    # Test metric return type
    @test m(A,b,x) isa Bool                                 
    i = progress(i,ntest,prefix)

    # Test broadcasted metric return type
    @test br_m(A,b,x) isa BitVector                         
    i = progress(i,ntest,prefix)

    # Test ksparse(1) on the null vector
    @test m(A,b,xz) == false                                
    i = progress(i,ntest,prefix)

    # Test ksparse([1,2,3]) on the null vector
    @test br_m(A,b,xz) == [false,false,false]               
    i = progress(i,ntest,prefix)

    # Test ksparse(1) on a one non-null entry vector
    xz[1] = 1.0
    @test m(A,b,xz) == true 
    i = progress(i,ntest,prefix)

    # Test ksparse([1,2,3]) on a one non-null entry vector
    @test br_m(A,b,xz) == [true,false,false]                
    i = progress(i,ntest,prefix)

    # Test ksparse(1) on a two non-null entry vector
    xz[2] = 2.0
    @test m(A,b,xz) == false                                
    i = progress(i,ntest,prefix)

    # Test ksparse([1,2,3]) on a two non-null entry vector
    @test br_m(A,b,xz) == [false,true,false]       
    i = progress(i,ntest,prefix)

    # Test ksparse(1) on a three non-null entry vector
    xz[3] = 3.0
    @test m(A,b,xz) == false                                
    i = progress(i,ntest,prefix)

    # Test ksparse([1,2,3]) on a three non-null entry vector
    @test br_m(A,b,xz) == [false,false,true]                
    i = progress(i,ntest,prefix,"\n")
end