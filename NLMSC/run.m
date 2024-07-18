clear all; close all;
addpath(genpath('.'));
load NGS.mat;X{1} = data{1}; X{2} = data{2};X{3} = data{3};classid = truelabel{1}';
for v =1:length(X)
   X{v} = NormalizeFea(X{v},1); 
%   X{v} = X{v}./repmat(sqrt(sum(X{v}.^2,1)),size(X{v},1),1); 
end
disp('Performing ...');
begintime=tic;
Para.para1 = 1e0;    
Para.para2 = 1e2; 
Para.para3 = 1000;
Para.k = length(unique(classid));
k = length(unique(classid));
iter = 1;
for i1 = 1:iter               
    tic
    [out_Z,out_C,outL,Converg_values]   = NLMSC(X,Para);
    toc
    niter = 20;
    n = size(out_Z{1},1);
    mv = size(out_Z,2);

    Z = eye(n); 
    Zv = repmat(Z,[1,1,3]);
    c = length(unique(classid));  
    wv = ones(mv,1)/mv;
    for v = 1:length(X)
        out_Z{v} = (out_Z{v}+out_Z{v}')/2;
    end
    disp(size(out_Z{v}))
    affinityMatrix = 0;
    for ii = 1:niter
        disp(['now this is the ',num2str(ii),' th iter']);
         Z = (Z+Z')/2; 
         for i=1:mv
            Zv(:,:,i) = out_Z{i};
            T = Zv(:,:,i);
            [T,~] = closest_neighbors(T,300,500);  
            T = (T+T')/2; 
            Zv(:,:,i) = T; 
            wv(i) = 1/2/norm(Zv(:,:,i)-Z,'fro'); 
            M(:,:,i) = wv(i)*Zv(:,:,i); 
         end
        for ij = 1:n
            Z(:,ij) = (sum(M(:,ij,:),3))/sum(wv);
        end
        Z = Z - diag(diag(Z));
        result_idx = spectral_clustering(Z, k);
        result_idx = double(result_idx);
        disp(' is finished.');
        [f_score,precision,recall] = compute_f(classid,result_idx);
        [~, cr,~, ~, ~] = CalClassificationRate(classid,result_idx);
        NMI = nmi(classid,result_idx);
        [acc,~,~ ] = CalcMetrics( classid,result_idx);
        [AR,RI,~,~] = RandIndex(classid,result_idx);
        disp([acc, NMI,f_score,precision,recall]) 
    end 
end

