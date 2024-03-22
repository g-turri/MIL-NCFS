clc
clear 
close all

%% load data
curd = pwd;
addpath(strcat(curd,'/src'));
curdir=strcat(curd,'/data/Musk1');
curdir1=strcat(curd,'/data/Musk1/all.txt');
curdir2=strcat(curd,'/data/Musk1/molecule_num.txt');

all_molecules=importdata(curdir1);
molecule_num=importdata(curdir2);

mole_num=92;
pos=47;
neg=45;
instances=sum(molecule_num);
pos_instances=sum(molecule_num(1:pos));
neg_instances=instances-pos_instances;
dim=166;
metric = 'euclidean';
	
pos_fold=[5 5 5 5 5 5 5 4 4 4];
neg_fold=[5 5 5 5 5 4 4 4 4 4];

refs= 2;
citers= refs+2;
str = strcat('BCKNN_NCFS_R',num2str(refs),'_C',num2str(citers),'_Musk1');

sigma=1;
bestLambda=0;
%lambda=[0 0.01 0.05 0.1];
% alpha=1;
% eta=10^-6;
% iter=100;
rng(0)
%rand('state',sum(100*clock));
per_pos=randperm(pos);
%rand('state',sum(100*clock));
per_neg=randperm(neg);

for fold=1:10
fval_old=0;
w_old=ones(dim,1);
fgrad=zeros(dim,1);
%Obj=zeros(iter,1);


        disp(fold)
        test_pos=pos_fold(fold);
        test_neg=neg_fold(fold);
        train_pos=pos-test_pos;
        train_neg=neg-test_neg;
        info=[train_pos,train_neg,test_pos,test_neg];
        
        trainIn=[];
        trainOut=[];
        testIn=[];
        testOut=[];
        
        trainInP=[];
        trainOutP=[];
        testInP=[];
        testOutP=[];
        
        trainInN=[];
        trainOutN=[];
        testInN=[];
        testOutN=[];
        
        trainIn=[];
        trainOut=[];
        testIn=[];
        testOut=[];
        countptr=0;
        countpte=0;
        countntr=0;
        countnte=0;
        
        DD_testVal_Neg=[];
        DD_testVal_Pos=[];
        DD_testVal=[];
        
        BagPosTR=cell(1,1);
        BagPosTE=cell(1,1);
        BagNegTR=cell(1,1);
        BagNegTE=cell(1,1);    
        
        for i=1:sum(pos_fold(1:(fold-1)))
            countptr=countptr+1;
            cur_mole=per_pos(i);
            base_pointer=sum(molecule_num(1:(cur_mole-1)));
            trainInP=[trainInP;all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:)];
            trainOutP=[trainOutP,molecule_num(cur_mole)];
            BagPosTR{countptr,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:);
        end
        
        for i=1:pos_fold(fold)
            countpte=countpte+1;
            cur_mole=per_pos(sum(pos_fold(1:(fold-1)))+i);
            base_pointer=sum(molecule_num(1:(cur_mole-1)));
            testInP=[testInP;all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:)];
            testOutP=[testOutP,molecule_num(cur_mole)];
            BagPosTE{countpte,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:);
        end
        
        for i=1:(sum(pos_fold)-sum(pos_fold(1:fold)))
            countptr=countptr+1;
            cur_mole=per_pos(sum(pos_fold(1:fold))+i);
            base_pointer=sum(molecule_num(1:(cur_mole-1)));
            trainInP=[trainInP;all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:)];
            trainOutP=[trainOutP,molecule_num(cur_mole)];
            BagPosTR{countptr,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(cur_mole)),:);
        end
        
        for i=1:sum(neg_fold(1:(fold-1)))
            countntr=countntr+1;
            cur_mole=per_neg(i);
            base_pointer=sum(molecule_num(1:(pos+cur_mole-1)));
            trainInN=[trainInN;all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:)];
            trainOutN=[trainOutN,molecule_num(pos+cur_mole)];
            BagNegTR{countntr,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:);
            trainOut=[trainOut,molecule_num(pos+cur_mole)];
        end        
            
        for i=1:neg_fold(fold)
            countnte=countnte+1;
            cur_mole=per_neg(sum(neg_fold(1:(fold-1)))+i);
            base_pointer=sum(molecule_num(1:(pos+cur_mole-1)));
            testInN=[testInN;all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:)];
            testOutN=[testOutN,molecule_num(pos+cur_mole)];
            BagNegTE{countnte,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:);
        end        
            
        for i=1:(sum(neg_fold)-sum(neg_fold(1:fold)))
            countntr=countntr+1;
            cur_mole=per_neg(sum(neg_fold(1:fold))+i);
            base_pointer=sum(molecule_num(1:(pos+cur_mole-1)));
            trainInN=[trainInN;all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:)];
            trainOutN=[trainOutN,molecule_num(pos+cur_mole)];
            BagNegTR{countntr,1}=all_molecules((base_pointer+1):(base_pointer+molecule_num(pos+cur_mole)),:);
        end  
        
        
            Neg=size(BagPosTR,1);
            Pos=size(BagNegTR,1);
            N=Neg+Pos;
            obs_weights=ones(1,N);
            NegTE=size(BagNegTE,1);
            PosTE=size(BagPosTE,1);
            ntest=NegTE+PosTE;
            
            BagTE=[BagNegTE;BagPosTE];
            BagTR=[BagNegTR; BagPosTR];

            lab_train=[ones(size(BagNegTR,1),1); 2*ones(size(BagPosTR,1),1)];
            lab_test=[ones(size(BagNegTE,1),1); 2*ones(size(BagPosTE,1),1)];
%             tempvec=[];   
%             for ii=1:numel(BagTR)
%                 temp=size(BagTR{ii,1},1);
%                 tempvec=[tempvec; temp];
%             end
%              tempvec2=[];   
%             for ii=1:numel(BagTE)
%                 temp=size(BagTE{ii,1},1);
%                 tempvec2=[tempvec2; temp];
%             end
            
%             Xtr = cell2mat(BagTR);  
%             Xte=cell2mat(BagTE);
%             [Xtrain, mu, sigma1] = zscore(Xtr);
%             %sigma1(sigma1==0)=eps;
%             BagTRnorm = mat2cell(Xtrain,tempvec, dim);
%             BagTRNegNorm=[];
%             BagTRPosNorm=[];
%             tempvec3=[];
%             tempvec4=[];
%             for gi=1:Neg
%                 BagTRNegNorm=[BagTRNegNorm; BagTRnorm{gi,1}];
%                 temp=size(BagTRnorm{gi,1},1);
%                 tempvec3=[tempvec3; temp];
%             end
%             
%             for gi=1:Pos
%                 BagTRPosNorm=[BagTRPosNorm; BagTRnorm{Neg+gi,1}];
%                 temp=size(BagTRnorm{Neg+gi,1},1);
%                 tempvec4=[tempvec4; temp];
%             end
%             BagTRNegNorm2=mat2cell(BagTRNegNorm,tempvec3, dim);
%             BagTRPosNorm2=mat2cell(BagTRPosNorm,tempvec4, dim);

            
%             C = bsxfun(@minus,Xte,mu);
%             Xte2 = bsxfun(@rdivide,C,sigma1);
%             BagTEnorm = mat2cell(Xte2,tempvec2, dim);
            
        solver = Solver(N);
        solver.NumComponents                = N;
        solver.SolverName                   = 'lbfgs';
        solver.HessianHistorySize           = 15;
        solver.InitialStepSize              = [];
        solver.LineSearchMethod             = 'weakwolfe';
        solver.MaxLineSearchIterations      = 20;
        solver.GradientTolerance            = 10^-6;
        solver.InitialLearningRate          = [];
        solver.MiniBatchSize                = 10;
        solver.PassLimit                    = 5;
        solver.NumPrint                     = 10;
        solver.NumTuningIterations          = 20;
        solver.TuningSubsetSize             = 100;
        solver.IterationLimit               = 100;
        solver.StepTolerance                = 10^-6;
        solver.MiniBatchLBFGSIterations     = 10;
        solver.InitialLearningRateForTuning = 0.1;
        solver.ModificationFactorForTuning  = 2;
        solver.HaveGradient                 = true;
        solver.Verbose                      = 0;
        
        % Minimization
        w=ones(dim,1);
        fun=@(w)myfun_w(w,BagTR,lab_train,sigma,bestLambda,obs_weights,metric);
        results=doMinimization(solver,fun,w,N);
        w=results.xHat;

        %% Prediction
        wsquared=w.^2;
        w2{fold} = wsquared;
        
        if Pos>Neg
            tie=1;
        else
            tie=0;
        end
        
        [predictNCA{fold}] = BCKNNw(BagPosTR,BagNegTR,BagTE,refs,citers,tie,wsquared,metric);
        [predict{fold}] = BCKNNw(BagPosTR,BagNegTR,BagTE,refs,citers,tie,ones(dim,1),metric);
    
    accuracyNCA_CV(fold)=mean((predictNCA{fold} == lab_test -1));
    accuracy_CV(fold) =mean(predict{fold} == lab_test -1);    
    macroNCA(fold) = my_micro_macro(predictNCA{fold},lab_test - 1)
    macro(fold) = my_micro_macro(predict{fold},lab_test - 1)


end

% delete(gcp('nocreate'))

disp('Accuracy NCA')
disp(mean(accuracyNCA_CV));

disp('MacroF1 NCA')
disp(mean(macroNCA));

disp('Accuracy')
disp(mean(accuracy_CV));

disp('MacroF1')
disp(mean(macro));

save(str)