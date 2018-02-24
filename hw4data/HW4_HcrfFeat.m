classdef HW4_HcrfFeat
% Class for features for HCRF    
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 01-Apr-2016
% Last modified: 01-Apr-2016    
    properties
        nState_;
        nClass_
        nDim_;      
        phi_s_rowIdxs;
        phi_s_colIdxs;
        phi_lt;   
        phi1_s_rowIdxs;
        phi1_s_colIdxs;
        phi1_lt;
    end
    
    methods        
        function obj = HW4_HcrfFeat(nState, nClass, nDim)
            obj.nState_ = nState;
            obj.nClass_ = nClass;
            obj.nDim_  = nDim;   
                        
            XtBlck = repmat((1:nDim)', 1, nState);            
            XtBlcks = repmat({XtBlck}, 1, nState);
            phi_s = sparse(blkdiag(XtBlcks{:}));
            
            [obj.phi_s_rowIdxs, obj.phi_s_colIdxs] = find(phi_s);                
            
            s_t = repmat(1:nState, nState,1);
            s_t1 = s_t';
            
            nState2 = nState*nState;
            colIdxs = 1:nState2;
            
            obj.phi_lt = cell(1, nClass);
            
            for y =1:nClass
                rowIdxs = s_t + (y-1)*nState;
                phi_l_y = sparse(rowIdxs, colIdxs, 1, nState*nClass, nState2);

                rowIdxs = s_t + (s_t1-1)*nState + (y-1)*nState*(nState+1);
                phi_t_y = sparse(rowIdxs, colIdxs, 1, nState*(nState+1)*nClass, nState2);                
                obj.phi_lt{y} = cat(1, phi_l_y, phi_t_y);
            end
            
            % For special case t=1
            XtBlcks = repmat({(1:nDim)'}, 1, nState);             
            phi_s = sparse(blkdiag(XtBlcks{:}));
            [obj.phi1_s_rowIdxs, obj.phi1_s_colIdxs] = find(phi_s);
            
            s_t = 1:nState;
            for y=1:nClass
                rowIdxs = s_t + (y-1)*nState;
                phi1_l_y = sparse(rowIdxs, s_t, 1, nState*nClass, nState);

                rowIdxs = s_t + nState*nState + (y-1)*nState*(nState+1);
                phi1_t_y = sparse(rowIdxs, s_t, 1, nState*(nState+1)*nClass, nState);
                obj.phi1_lt{y} = cat(1, phi1_l_y, phi1_t_y);
            end
        end
                
        % Compute multiple feature vectors phi_t for all s_t and s_{t-1}
        % We use s_t1 to denote s_{t-1}
        % Output: 
        %   If t > 1
        %       featVecs: 2D matrix with (nState*nState) columns
        %       featVecs(:, s_t1 + (s_t - 1)*nState) = phi_t(y, s_t, s_t1, X)        
        %   If t = 1
        %       featVecs: 2D matrix with nState columns
        %       featVecs(:, s_1) = phi_1(y, s_1, s_0, X)        
        function featVecs = cmpFeatVecs(self, X, y, t)
            nState = self.nState_;
            
            if t > 1                     
                Xt2 = repmat(X(:,t), nState*nState, 1);
                phi_s = sparse(self.phi_s_rowIdxs, self.phi_s_colIdxs, Xt2(:));                
                featVecs = cat(1, phi_s, self.phi_lt{y});
            else
                Xt2 = repmat(X(:,t), nState,1);
                phi_s = sparse(self.phi1_s_rowIdxs, self.phi1_s_colIdxs, Xt2(:));
                featVecs = cat(1, phi_s, self.phi1_lt{y});
            end
        end;
        
        % Compute multiple feature vectors phi_t for all s_t and s_t1
        % This is exactly the same as cmpFeatVecs, but much slower. 
        % However, it shows clearly how the feature vectors are computed
        function featVecs = cmpFeatVecs_slow(self, X, y, t)
            nState = self.nState_;
            if t > 1
                featVecs = cell(nState, nState);
                for s_t1=1:nState
                    for s_t = 1:nState
                        featVecs{s_t1, s_t} = self.cmpFeatVec_slow(X, y, t, s_t, s_t1);                
                    end;
                end                
            elseif t == 1
                featVecs = cell(1, nState);
                for s_t = 1:nState
                    featVecs{s_t} = self.cmpFeatVec_slow(X, y, t, s_t, []); 
                end;
            end;
            featVecs = cat(2, featVecs{:});                
        end;

        
        % Compute feature vector: phi(y, s_t, s_t1, X)
        % return a column vector for the feature        
        function featVec = cmpFeatVec_slow(self, X, y, t, s_t, s_t1)
            nDim   = self.nDim_;
            nClass = self.nClass_;
            nState = self.nState_;
            if t == 1
                s_t1 = nState + 1; % special value for s_0
            end;            
            
            phi_s = zeros(nDim, nState);            
            phi_s(:, s_t) = X(:,t);                                    
            phi_l = zeros(nState,nClass);
            phi_l(s_t, y)  = 1;
            phi_t = zeros(nState,nState+1,nClass);
            phi_t(s_t, s_t1, y) = 1;
            featVec = sparse([phi_s(:); phi_l(:); phi_t(:)]);
        end        
    end;   
    
    methods (Static)
        function compareSlowAndFast()
            X = rand(60, 200);
            nDim = size(X,1);
            nState = 12;
            nClass = 5;
                        
            featObj = HW4_HcrfFeat(nState, nClass, nDim);            
            featFunc = @featObj.cmpFeatVecs;
            featFunc_slow = @featObj.cmpFeatVecs_slow;
            
            nRun = size(X,2);
            diffs = zeros(1, nRun);
            durT1 = 0;
            durT2 = 0;
            for t=1:size(X,2)
                y = randi(nClass); % random class                
                startT1 = tic;                
                featVecs = featFunc(X, y, t);
                durT1 = durT1 + toc(startT1);
                
                startT2 = tic;
                featVecs_slow = featFunc_slow(X, y, t);
                durT2 = durT2 + toc(startT2);
                diffs(t) = sum(sum(abs(featVecs - featVecs_slow)));
            end;
            
            if sum(diffs) > 0
                error('Something is wrong. Slow and fast methods are not the same');
            else
                fprintf('Two methods return the same feature vectors\n');
            end;
            fprintf('Time, slow: %g, fast: %g\n', durT2, durT1);            
        end;        
    end;
end

