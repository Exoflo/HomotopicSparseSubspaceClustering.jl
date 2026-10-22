module HomotopicSparseSubspaceClustering
    # utils.jl
    using Arpack: eigs                              # Usage : spectral_clustering()
    using Hungarian: munkres,STAR                   # Usage : misclassification()
    using LinearAlgebra: I, Symmetric, normalize!   # Usage : error_lin_sys()
    using ParallelKMeans: kmeans, Hamerly           # Usage : spectral_clustering()
    using SparseArrays: SparseMatrixCSC, dropzeros  # Usage : build_adjacency!(), spectral_clustering(), _thr()
    using TSVD: tsvd                                # Usage : data_projection()

    # ADMM 
    using LinearAlgebra: diagind                    # Usage : admm_lasso_lin(), admm_lasso_affine(), admm_outliers_lin(), admm_outliers_affine()
    using Distances: opnorm                         # Usage : admm_outliers_lin(), admm_outliers_affine()

    # SSC.jl / HSSC.jl
    using SparseArrays: sparse                      # Usage : ssc(), hssc(), knn_hssc(), broadcasted_hssc(), broadcasted_knn_hssc()

    # homotopy.jl
    using Distances: pairwise,Euclidean             # Usage : knn_homotopy_solver() - broadcasted_knn_homotopy_solver()
    using LinearAlgebra: pinv                       # Usage : homotopy() - broadcasted_homotopy()

    # metrics.jl 
    using LinearAlgebra: norm                       # Usage : relerror()

    include("utils.jl")
    include("ADMM.jl")
    include("SSC.jl")
    include("homotopy.jl")
    include("metrics.jl")
    include("HSSC.jl")

    export ssc
    export test_metric, test_br_metric, ksparse, relerror
    export hssc, knn_hssc, broadcasted_hssc, broadcasted_knn_hssc
    # export khssc, ehssc, knn_khssc, knn_ehssc, broadcasted_khssc, broadcasted_ehssc, broadcasted_knn_khssc, broadcasted_knn_ehssc
    export misclassification
end
