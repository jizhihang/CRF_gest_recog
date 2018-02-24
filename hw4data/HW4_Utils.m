classdef HW4_Utils
% Utility functions for HW4 of ML Class Spring 2016    
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 01-Apr-2016
% Last modified: 01-Apr-2016    
    
    methods (Static)
        % Log Sum Exponential Trick
        % A: k*n matrix
        % lse = 1*n vector, lse = log(sum(exp(A), 1)).
        function lse = logSumExp(A)
            maxA = max(A,[], 1);
            infIdxs = isinf(maxA);            
            if any(infIdxs)
                notInfIdxs = ~infIdxs;
                maxA2 = maxA(notInfIdxs);
                A = A(:,notInfIdxs) - repmat(maxA2, [size(A,1), 1]);
                lse = zeros(1, length(notInfIdxs));            
                lse(infIdxs) = maxA(infIdxs);
                lse(notInfIdxs) = maxA2 + log(sum(exp(A),1));
            else
                A  = A - repmat(maxA, [size(A,1), 1]);                
                lse = maxA + log(sum(exp(A),1));
            end
        end
        
        % From Unnormalized Log probability to Normalized Probablity, using Log-Sum-Exp trick
        % Input:
        %   A: d*n matrix, each column is the log of unnormalize  probabilities
        % Output
        %   B: B(:,i) is the normalized log probability corresspond to unnormalize prob A(:,i)
        %   B(:,i) has the property: sum(exp(B(:,i)) = 1.         
        %   In other words: 
        %       B = exp(A);
        %       B = B./repmat(sum(B,1), size(A,1), 1); 
        %       But this function uses the log-sum-exponential trick to avoid numerical problem
        function B = logUnnormProb2NormProb(A)
            lse = HW4_Utils.logSumExp(A);
            B = exp(A - repmat(lse, size(A,1), 1));
        end;

    end    
end

