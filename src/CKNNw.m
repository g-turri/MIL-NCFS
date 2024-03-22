function [labels]=CKNNw(PBags,NBags,testBags,Refs,Citers,indicator,w,metric)
%  CKNN  Using the Citation KNN algorithm[1] to get the labels for bags in testBags, where minmum Hausdorff distance is used to measure the distances between bags
%     CKNN takes,
%        PBags     - an Mx1 cell array where the jth instance of ith positive bag is stored in PBags{i}(j,:)
%        NBags     - an Nx1 cell array where the jth instance of ith negative bag is stored in NBags{i}(j,:)
%        testBags  - a Kx1 cell array where the jth instance of ith test bag is stored in testBags{i}(j,:)
%         Refs     - the number of referecnes for each test bag
%        Citers    - the number of citers for each test bag
%        indicator - when the nearest neighbours have equal number of postive bags and negative bags, if indicator==1,then the output
%                    label is 1, otherwise 0, default=0
%
%     and returns,
%        labels    - a Kx1 vector which correspond to the output label of each bag in testBags
%
% For more details, please reference to bibliography [1]
% [1] J. Wang and J.-D. Zucker. Solving the multiple-instance problem: a lazy learning approach. In: Proceedings of the 17th
%     International Conference on Machine Learning, San Francisco, CA: Morgan Kaufmann, 1119-1125, 2000.

if(nargin<=4)
    error('not enough input parameters');
end

if(nargin<=5)
    indicator=0;
end

size1=size(PBags);
size2=size(NBags);
size3=size(testBags);
num_pbags=size1(1);
num_nbags=size2(1);
num_testbags=size3(1);
if((Refs>num_pbags+num_nbags)|(Citers>num_pbags+num_nbags))
    erorr('too many Refs or Citers');
end

labels=zeros(num_testbags,1);
for i=1:num_testbags
    num_bags=num_pbags+num_nbags+1;
    dist=-eye(num_bags);   %for every bag in testBAgs, initialize the distance matrix
    for j=1:num_bags
        if(j==1)
            for k=(j+1):(num_pbags+1)
                [~,dist(j,k)]=minHausdorffW(testBags{i},PBags{k-1},w',metric);
                dist(k,j)=dist(j,k);
            end
            for k=(num_pbags+2):num_bags
                [~,dist(j,k)]=minHausdorffW(testBags{i},NBags{k-num_pbags-1},w',metric);
                dist(k,j)=dist(j,k);
            end
        else
            if((j>=2)&(j<=num_pbags+1))
                for k=(j+1):(num_pbags+1)
                    [~,dist(j,k)]=minHausdorffW(PBags{j-1},PBags{k-1},w',metric);
                    dist(k,j)=dist(j,k);
                end
                for k=(num_pbags+2):num_bags
                    [~,dist(j,k)]=minHausdorffW(PBags{j-1},NBags{k-num_pbags-1},w',metric);
                    dist(k,j)=dist(j,k);
                end
            else
                for k=(j+1):num_bags
                    [~,dist(j,k)]=minHausdorffW(NBags{j-num_pbags-1},NBags{k-num_pbags-1},w',metric);
                    dist(k,j)=dist(j,k);
                end
            end
        end
    end
    
    rp=0;   %get the references and citers of current test bag
    rn=0;
    cp=0;
    cn=0;
    
    [temp,index]=sort(dist(1,:));
    for ref=1:Refs
        if(index(ref+1)<=num_pbags+1)
            rp=rp+1;
        else
            rn=rn+1;
        end
    end
    
    for pointer=2:(num_pbags+1)
        [temp,index]=sort(dist(pointer,:));
        if(find(index==1)<=Citers+1)
            cp=cp+1;
        end
    end
    for pointer=(num_pbags+2):num_bags
        [temp,index]=sort(dist(pointer,:));
        if(find(index==1)<=Citers+1)
            cn=cn+1;
        end
    end

    pos=rp+cp;
    neg=rn+cn;

    %post_prob = [pos/(pos+neg) neg/(pos+neg)];
    
    if(pos>neg)
        labels(i,1)=1;
    else
        if(pos==neg)
            if(indicator==1)
                labels(i,1)=1;
            end
        end
        labels(i,1)=0;
    end
end


