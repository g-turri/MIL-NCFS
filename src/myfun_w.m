function [fval,fgrad]=myfun_w(w,X,y,sigma,lambda,obs_weights,metric)

[N,P]=size(X);
fgrad=zeros(P,1);

wsquared=w.^2;
fval=0;
for i=1:N
    xi=X{i,1};
    yi=y(i);
    
    for in=1:N
        %P-by-N matrix distance
        [d(in,:),dw(in)]=minHausdorffW(X{in,1},xi,wsquared.',metric);
    end
    
    % 1-by-N vector weighted distance
    dw(i)=inf; % dw to Inf since we are not interested in the distance of it and itself.
    dw=dw-min(dw); % ensures that sum(pij) is not zero.
    
    % 1-by-N vector of probabilities pij.
    pij=obs_weights.*exp(-dw/sigma);
    pij=pij/sum(pij); % normalization
    
    yij=-double(bsxfun(@eq,yi,y')); % - for minimization
    yijpij=yij.*pij;
    
    pi=sum(yijpij);
    
    %Update
    fval=fval+pi;
    fgrad=fgrad+sum(bsxfun(@times,pi*pij-yijpij,d.'),2);
end
% Update fval fgrad
fval=fval/N;
fgrad=((2*w/sigma).*fgrad)/N;

fval=fval+lambda*sum(wsquared); % Add contribution of the regularization term.
fgrad=fgrad+2*lambda*w; % Add contribution of the regularization term.