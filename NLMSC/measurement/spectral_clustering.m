function [ ids ] = spectral_clustering(W, k)
    D = diag(1./(eps+sqrt(sum(W, 2))));
    W = D * W * D;
    [U, s, V] = svd(W);
    V = U(:, 1 : k);
    V = normr(V);
    %ids=litekmeans(V, k,  'Replicates', 500);
    %ids = kmeans(V, k, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
    ids = kmeans(V, k, 'start','sample','maxiter', 1000,'replicates',100,'EmptyAction','singleton');
    
    disp("spectral_clustering")
    disp(length(unique(ids)))
    disp(k)

end
