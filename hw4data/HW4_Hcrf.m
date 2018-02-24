classdef HW4_Hcrf
% Homework for Linear Chain Hidden CRF for ML class Spring 2016
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 08-Feb-2016
% Last modified: 08-Feb-2016    
    
    methods (Static)
        
        function [trac,valac,conf]=main()
            [trD, trLb, valD, valLb, tstD] = MHW_ChaLearnData.load3ClassData();
                        
            nState = 10;            
            nClass = max(trLb(:));
            nDim = size(trD{1},1);
            featObj = HW4_HcrfFeat(nState, nClass, nDim);
            featFunc = @featObj.cmpFeatVecs;            
            lambda=0.001;
            
            % Call your trainning code
            % w = HW4_Hcrf.train(valD, valLb, featFunc, nState, nClass, nDim, lambda);            
            
            % Your evaluation code
            % predLb = HW4_Hcrf.predict(tstD, featFunc, w, nState)
            
            % Code to evaluate the prediction and compute confusion matrix
            %[fVal,grad] = HW4_Hcrf.cmpObjAndGrad(valD, valLb, featFunc, nClass, nState, w, lambda)
            [trac,valac,conf]=HW4_Hcrf.four(trD,trLb,valD,valLb,tstD,featFunc,nState,nClass,nDim,lambda);
        end;

        % You need to implement this function
        function w = train(Ds, lb, featFunc, nState, nClass, nDim, lambda)  
            fun = @(w)HW4_Hcrf.cmpObjAndGrad(Ds, lb, featFunc, nClass, nState, w, lambda);
            options=optimoptions(@fminunc,'Algorithm','quasi-newton','Display','iter-detailed','MaxIter',50,'GradObj','on');
            w=rand(610,1);
            w =fminunc(fun,w,options);
        end;
        
        function [trac,valac,conf]=four(trD,trLb,valD,valLb,tstD,featFunc,nState,nClass,nDim,lambda)
            w = HW4_Hcrf.train(trD, trLb, featFunc, nState, nClass, nDim, lambda);
            predLb = HW4_Hcrf.predict(trD, featFunc, w, nState, nClass);
            trac=sum((trLb-predLb)==0)/size(trLb,1);
           % w = HW4_Hcrf.train(valD, vallb, featFunc, nState, nClass, nDim, lambda);
            predLb = HW4_Hcrf.predict(valD, featFunc, w, nState, nClass);
            predLbs = HW4_Hcrf.predict(tstD, featFunc, w, nState, nClass);
            valac = sum((valLb-predLb)==0)/size(valLb,1);
            conf=zeros(3,3);
            conf(1,1)=sum(((valLb==1)&(predLb==1)));
            conf(1,2)=sum(((valLb==1)&(predLb==2)));
            conf(1,3)=sum(((valLb==1)&(predLb==3)));
            conf(2,1)=sum(((valLb==2)&(predLb==1)));
            conf(2,2)=sum(((valLb==2)&(predLb==2)));
            conf(2,3)=sum(((valLb==2)&(predLb==3)));
            conf(3,1)=sum(((valLb==3)&(predLb==1)));
            conf(3,2)=sum(((valLb==3)&(predLb==2)));
            conf(3,3)=sum(((valLb==3)&(predLb==3)));
           answer = [(1:529)',predLbs];
            csvwrite('predTestLabels_3classes.csv',answer);
        end
        % You need to implement this function
        function predLb = predict(tstD, featFunc, w, nState, nClass)
            p=zeros(nClass,1);
            predLb=zeros(size(tstD,1),1);
            for i=1:size(tstD,1)
            for j=1:nClass
            [logAlphas, logBetas] = HW4_Hcrf.forwardBackward(tstD{i}, j, w, featFunc, nState);
            
            sum=HW4_Utils.logSumExp(logAlphas+logBetas);
            p(j,1)=sum(:,end);
            end
            norm=HW4_Utils.logSumExp(p);
            [~,predLb(i,1)]=max(p-norm);
            end
           
        end
        
        % Implement this function for training loss, 
        % it should return the function value and gradient wrt w
        function [fVal,grad] = cmpObjAndGrad(Ds, lb, featFunc, nClass, nState, w, lambda) 
            sumtotal=0;
            sumgtotal=zeros(size(w,1),1);
            for i=1:size(Ds,1)
                [logAlphas, logBetas] = HW4_Hcrf.forwardBackward(Ds{i}, lb(i,1), w, featFunc, nState);
                derVec = HW4_Hcrf.cmpDerOfLogZ(Ds{i}, lb(i,1), w, featFunc, nState, logAlphas, logBetas);
                sumg=zeros(size(derVec,1),1);
                sum = HW4_Utils.logSumExp(logAlphas+logBetas);
                sum = sum(:,end);
                p=zeros(nClass,1);
                C=cell(nClass,2);
                for j=1:nClass
                    if j==lb(i,1)
                        p(j,:)=sum;
                    else
                    [logAlphas, logBetas] = HW4_Hcrf.forwardBackward(Ds{i}, j, w, featFunc, nState);
                    temp=HW4_Utils.logSumExp(logAlphas+logBetas);
                    p(j,1)=temp(:,end);
                    C{j,1}=logAlphas;
                    C{j,2}=logBetas;
                    end
                end
                norm=HW4_Utils.logSumExp(p);
                sum=sum-norm;
                for j=1:nClass
                    if j==lb(i,1)
                    sumg = sumg+derVec*exp(sum);
                    else
                    
                    sumg = sumg+HW4_Hcrf.cmpDerOfLogZ(Ds{i}, j, w, featFunc, nState, C{j,1}, C{j,2})*exp(p(j,1)-norm);
                    end
                end
              sumtotal=sumtotal+sum;
              sumgtotal=sumgtotal+derVec-sumg;
           end
            fVal=lambda*w'*w/2-sumtotal/300;
            grad=lambda*w-sumgtotal/300;
        end;
        
        % Compute d(log Z(y,X))/dw
        % X: d*seqLen matrix for a time series
        % y: a scalar for class label
        % w: current weight vector
        % featFunc: a handler to a feature function
        % nState: number of hidden states        
        % logAlphas, logBetas: nState*seqLen matrixes
        % Outputs:
        %   derVec: derivative vector = d(log Z(y,X))/dw
        function derVec = cmpDerOfLogZ(X, y, w, featFunc, nState, logAlphas, logBetas)
            seqLen  = size(X,2);
                        
            featVecs = featFunc(X, y, 1);
            logUnnormP = logAlphas(:,end);
            normP = HW4_Utils.logUnnormProb2NormProb(logUnnormP(:));
            derVec = featVecs*normP;
            
            for t=2:seqLen
                featVecs = featFunc(X, y, t); 
                wFeats = w'*featVecs;
                wFeats = reshape(wFeats, nState, nState);
                logUnnormP = repmat(logAlphas(:,t-1), 1, nState) + ...
                    wFeats + repmat(logBetas(:,t)', nState, 1);
                                
                normP = HW4_Utils.logUnnormProb2NormProb(logUnnormP(:));
                derVec =  derVec + featVecs*normP;
            end
        end;

        
        % Forward-backward algorithm
        % X: d*seqLen matrix for a time series
        % y: a scalar for class label
        % w: current weight vector
        % featFunc: a handler to a feature function
        % nState: number of hidden states
        % Outputs:
        %   logAlphas, logBetas: nState*seqLen matrixes
        %   logAlphas(i,t) = log(alpha_t(X_t = i));
        function [logAlphas, logBetas] = forwardBackward(X, y, w, featFunc, nState)
            seqLen = size(X,2);
            [logAlphas, logBetas] = deal(zeros(nState, seqLen));
            
            %forward pass
            featVecs = featFunc(X, y, 1);
            logAlphas(:,1) = w'*featVecs;
            for t=2:seqLen                
                featVecs = featFunc(X, y, t);                
                wFeats = reshape(w'*featVecs, nState, nState);                                
                A = wFeats + repmat(logAlphas(:,t-1), 1, nState);                
                logAlphas(:,t) = HW4_Utils.logSumExp(A);                
            end;
            
            if nargout > 1 %backward pass if necessary                
                logBetas(:, seqLen) = 0;
                for t=seqLen-1:-1:1                    
                    featVecs = featFunc(X, y, t+1);                    
                    wFeats = reshape(w'*featVecs, nState, nState);                    
                    A = wFeats' + repmat(logBetas(:,t+1), 1, nState);
                    logBetas(:,t) = HW4_Utils.logSumExp(A);
                end;
            end
        end;
        
        % A demo function that show:
        % 1. How to construct the feature function
        % 2. How forward-back algorithm is called
        % 3. Test that computation of derivative is correct
        function test_cmpDerOfLogZ()
            X = rand(60, 70); % random data, nClass, nState, nDim
            nDim = size(X,1);
            nState = 12;
            nClass = 5;
            y = 2;
                        
            featObj = HW4_HcrfFeat(nState, nClass, nDim);            
            featFunc = @featObj.cmpFeatVecs;            
            
            % let's figure out the dim of the feature vector
            % by calling the function and check the dimension
            featVecs = featFunc(X, 1, 1);
            wDim = size(featVecs,1);
            
            w1 = rand(wDim,1);
            [logAlphas, logBetas] = HW4_Hcrf.forwardBackward(X, y, w1, featFunc, nState);
            f1 = HW4_Utils.logSumExp(logAlphas(:,end)); % function value at w1
            
            % The derivative computed analytically
            dw1 = HW4_Hcrf.cmpDerOfLogZ(X, y, w1, featFunc, nState, logAlphas, logBetas);

            % Small deviation vector
            epsVec = 1e-2*rand(size(w1));
            
            w2 = w1 + epsVec;            
            logAlphas2 = HW4_Hcrf.forwardBackward(X, y, w2, featFunc, nState);
            f2 = HW4_Utils.logSumExp(logAlphas2(:,end)); % function value at w2
            
            f2b = f1 + dw1'*epsVec;
            fprintf('If the derivative function is correct f2 and f2b should be similar\n');
            fprintf('f1: %f, f2b: %f, f2: %f\n', f1, f2b, f2);           
        end

    end    
end

