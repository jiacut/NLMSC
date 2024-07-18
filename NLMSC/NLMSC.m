function [Z,C,L,Converg_values] = NLMSC(inData,inPara)
    X = inData;
    alpha1 = inPara.para1;
    alpha2 = inPara.para2;
    rankBound = inPara.para3;
    numView = length(X);
    numSamp = size(X{1},2);
    %% Initialization (L{v}, C{v},Y{v})
    for v = 1: numView
        numDim(v) = size(X{v},1);
        XvX{v} = X{v}'*X{v};
        XXv{v} = X{v}*X{v}';
        L{v}  = eye(numSamp,rankBound);
        C{v}  = eye(rankBound,numSamp);
        Y{v}  = zeros(numSamp,numSamp);  
    end

    tol = 1e-2;
    mu = 1e-3;
    max_mu = 1e3;
    max_iter = 50;
    rho = 1.3;
    isConverg = 0;
    iter = 0;
    Converg_values = zeros(max_iter,1);

    %% Optimization
    while(isConverg == 0)
        iter = iter + 1;
        fprintf('----------- The  %d processing iteration -------------\n',iter);
        %% Update Z{v}
        for v = 1:numView
            if numSamp < numDim(v)
                tempZ = (XvX{v} + mu*eye(numSamp,numSamp))\eye(numSamp,numSamp);
            else 
                tempZ = (1/mu)*eye(numSamp,numSamp) -  (1/(mu^2))*X{v}'*inv(eye(numDim(v),numDim(v)) + (1/mu)* XXv{v})*X{v};
            end            
            Z{v} = tempZ*(XvX{v}-Y{v} + mu*L{v}*C{v});
          
        %% update L{v}
            R{v} = Z{v} + Y{v}/mu;
            tempL = C{v}* R{v}';
            [Ul,~,Vl] = svd(tempL,'econ');
            L{v} = Vl*Ul';
    
        %% update C{v}        
            tempC = mu*L{v}'*R{v} ;
            C{v}  = solve_SVT(tempC/(mu+alpha2),alpha1/(mu+alpha2));
        end
     %% update Y
        for v = 1:numView
            Y{v} = Y{v} + mu*(Z{v} - L{v}*C{v});
        end
        mu = min(rho*mu,max_mu);

        %% Check the convegence
        isConverg = 1;
        condition1 = 0;
     
        for v = 1:numView
            condition1 = condition1 + norm(Z{v} - L{v}*C{v},'fro'); 
        end
        Converg_values(iter) = condition1;
    
        if condition1 >= tol
            fprintf('tol1 = % 7.1f   \n', condition1);
            isConverg  = 0; 
        end
        
        if iter > max_iter
            isConverg = 1;
        end
    end
end

function J = solve_SVT(CC,beta)
    [U,sigma,VV] = svd(CC,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>beta));
    if svp>=1
        sigma = sigma(1:svp)-beta;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*VV(:,1:svp)';
end