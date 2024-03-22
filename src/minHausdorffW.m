function [d,dw]=minHausdorffW(Bag1,Bag2, ws, distance)
% minHausdorffW  Compute the minimum weighted Hausdorff distance between two bags Bag1 and Bag2
% minHausdorff takes,
%   Bag1 - one bag of instances
%   Bag2 - the other bag of instanes
%   ws - the weighting vector 
%   distance - the distance to compute
%   and returns,
%   the distance vector d - the minimum Hausdorff distance between Bag1 and Bag2
%   between each feature;
%   the minimum weighted distance dw - the minimum weighted L1-norm Hausdorff
%   distance between Bag1 and Bag2;
%   index of the instances id_instance - the index of the instances selected to compute dw;

    size1=size(Bag1);
    size2=size(Bag2);
    line_num1=size1(1);
    line_num2=size2(1);
    dist=zeros(line_num1,line_num2);
    
    switch distance
        case 'cityblock'
             for i=1:line_num1
                for j=1:line_num2
                    A = (Bag1(i,:)-Bag2(j,:));
                    dist_temp{i,j}=abs(A);
                    dist(i,j)=sum(ws.*dist_temp{i,j});
                end
             end
             
        case 'euclidean'
             for i=1:line_num1
                for j=1:line_num2
                    A = (Bag1(i,:)-Bag2(j,:));
                    dist_temp{i,j}=A.^2;
                    dist(i,j)=sqrt(sum(ws.*dist_temp{i,j}));
                end
             end
            
        case 'chebychev' 
             for i=1:line_num1
                for j=1:line_num2
                    A = (Bag1(i,:)-Bag2(j,:));
                    dist_temp{i,j}=abs(A);
                    dist(i,j)=max(ws.*dist_temp{i,j});
                end
             end    
             
        case 'minkowski' 
             for i=1:line_num1
                for j=1:line_num2
                    A = (Bag1(i,:)-Bag2(j,:));
                    dist_temp{i,j}=A.^3;
                    dist(i,j)=(sum(ws.*dist_temp{i,j}))^(1/3);
                end
             end  
            
        otherwise
            error(strcat(distance,' distance is wrong typed or not implemented yet'))
    end

    dw=min(min(dist));
    [v,l]=min(dist(:));
%     [R, C]=ind2sub(size(dist),l);
%     d=dist_temp{R,C};
    d=dist_temp{l};
        
    