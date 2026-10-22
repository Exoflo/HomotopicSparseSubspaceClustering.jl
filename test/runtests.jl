using HomotopicSparseSubspaceClustering
using Test
using LinearAlgebra: I


include("test-HSSC.jl")
include("test-ksparse.jl")
include("test-misclassification.jl")
include("test-relerror.jl")
include("test-SSC.jl")

function progress(i::Integer,t::Integer,pre="",suf="")
    NMAX = 40;
    p = round(Int64,i/t * NMAX);
    print(pre,"| $(i == 0 ? " " : "")$(round(100*i/t,digits=2))% $("#"^p)$(" "^max(NMAX-p,0)) |",suf)
    return i+1
end

@testset "HomotopicSparseSubspaceClustering.jl" begin
    
    test_ssc()
    test_ksparse()
    test_relerror()
    test_hssc()
    test_misclassification()

    # Perform easy clustering
    A = [ones(3,3) zeros(3,3) zeros(3,3);zeros(3,3) ones(3,3) zeros(3,3);zeros(3,3) zeros(3,3) ones(3,3)];  
    print("Testing SSC results ... ")
    @test misclassification(ssc(A,nclust=3),[1,1,1,2,2,2,3,3,3],nclust=3)[1] == 0.0;   
    println("Done !")

    print("Testing HSSC results ... ")
    @test misclassification(hssc(A,ksparse(3),nclust=3),[1,1,1,2,2,2,3,3,3],nclust=3)[1] == 0.0;   
    println("Done !")

    println("")
end
