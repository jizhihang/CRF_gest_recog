classdef MHW_ChaLearnData
% Utility functions for loading and processing ChaLearn Data    
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 24-Jan-2016
% Last modified: 24-Jan-2016

    properties (Constant)
        dataDir = '../hw4data/ChaLearnDataset';        
    end
        
    methods (Static)      
        % maxTrPerClass: maximum training examples per class
        % Increase this if you want more training data. It can be set to inf for maximum number.
        function [trD, trLb, valD, valLb, tstD] =  load3ClassData(maxTrPerClass)
            classes = 5:7;
            if ~exist('maxTrPerClass', 'var')                
                maxTrPerClass = 100; 
            end
            [trD, trLb, valD, valLb, mD, PcaBasis] = MHW_ChaLearnData.loadData(classes, maxTrPerClass);
            load(sprintf('%s/tstData_3classes.mat', MHW_ChaLearnData.dataDir), 'tstD');
            
            tstD = MHW_ChaLearnData.normalizeData(tstD);
            tstD = MHW_ChaLearnData.applyPCA(tstD, mD, PcaBasis);             
            
        end;
        
        % maxTrPerClass: maximum training examples per class
        % Increase this if you want more training data. It can be set to inf for maximum number.
        function [trD, trLb, valD, valLb, tstD] = load20ClassData(maxTrPerClass)
            classes = 1:20;
            if ~exist('maxTrPerClass', 'var')                
                maxTrPerClass = 100; 
            end
            [trD, trLb, valD, valLb, mD, PcaBasis] = MHW_ChaLearnData.loadData(classes, maxTrPerClass);
            load(sprintf('%s/tstData_20classes.mat', MHW_ChaLearnData.dataDir), 'tstD');
            
            tstD = MHW_ChaLearnData.normalizeData(tstD);
            tstD = MHW_ChaLearnData.applyPCA(tstD, mD, PcaBasis);             
        end;
        
                % Display a random sequence
        function showRandom()
            load(sprintf('%s/trainData.mat', MHW_ChaLearnData.dataDir), 'trD');
            n = length(trD); % number of sequences
            k = randsample(n, 1);
            coords = reshape(trD{k}, [3, 20, size(trD{k},2)]); 
            MHW_ChaLearnData.drawSkt(coords);
        end;

        
        function [trD, trLb, valD, valLb, mD, PcaBasis] = loadData(classes, maxTrPerClass)
            load(sprintf('%s/trainData.mat', MHW_ChaLearnData.dataDir), 'trD');
            load(sprintf('%s/trainLabel.mat', MHW_ChaLearnData.dataDir), 'trLabel');
            
            load(sprintf('%s/valData.mat', MHW_ChaLearnData.dataDir), 'valD');
            load(sprintf('%s/valLabel.mat', MHW_ChaLearnData.dataDir), 'valLabel');
            
            if ~exist('maxTrPerClass')
                maxTrPerClass = inf;
            end;
            [trD, trLb]   = MHW_ChaLearnData.selData4Classes(trD, trLabel, classes, maxTrPerClass);
            [valD, valLb] = MHW_ChaLearnData.selData4Classes(valD, valLabel, classes, inf);
            
            % Normalize data
            trD = MHW_ChaLearnData.normalizeData(trD);
            valD = MHW_ChaLearnData.normalizeData(valD);
            
            % Learn PCA basis using training data
            [mD, PcaBasis] = MHW_ChaLearnData.learnPCA(trD);
            
            % Apply PCA to get lower dimensional data
            trD  = MHW_ChaLearnData.applyPCA(trD, mD, PcaBasis);
            valD = MHW_ChaLearnData.applyPCA(valD, mD, PcaBasis);             
        end
        

        % Ds: n*1 cell structure for n sequences
        function Ds = normalizeData(Ds)
            for i=1:length(Ds)
                D = Ds{i};                
                mD = mean(D, 2); % the action is translation invariant, so subtract the mean
                D = D - repmat(mD, 1, size(D,2));
                % Do L2 normalization
                l2Norm = sqrt(sum(D.^2,1));  
                D = D./repmat(l2Norm, size(D,1),1);
                
                Ds{i} = D;
            end
        end
        
        % Learn PCA basis and the mean vector
        function [mD, PcaBasis] = learnPCA(trD)
            % First get the mean vector and Pca basis
            D = cat(2, trD{:});
            mD = mean(D,2); % mean of the data
            cenD = D - repmat(mD, 1, size(D,2)); % centralize the data
            retainEnergy = 0.98; % PCA energy to retain
            PcaBasis = ml_pca(cenD, retainEnergy, 1);             
        end
        
        % Apply PCA to get lower dimensional data
        function LowDs = applyPCA(Ds, mD, PcaBasis)
            LowDs = cell(size(Ds));
            for i=1:length(Ds)
                LowDs{i} = PcaBasis'*(Ds{i} - repmat(mD, [1, size(Ds{i},2)]));
            end;
        end        
        
        % Select data for selected classes
        function [newD, newLb] = selData4Classes(D, lb, classes, maxPerClass)            
            [newD, newLb] = deal(cell(1, length(classes)));
            for i=1:length(classes);
                newD{i} = D(lb == classes(i));
                
                % Testing, take 100 examples only
                newD{i} = newD{i}(1:min(maxPerClass, length(newD{i}))); 
                
                newLb{i} = i*ones(length(newD{i}),1);
            end;
            newD = cat(1, newD{:});
            newLb = cat(1, newLb{:});            
        end;
        
        % the size of coords should be 3*20*t.
        function drawSkt(coords)        
            J=[20     1     2     1     8    10     2     9    11     3     4     7     7     5     6    14    15    16    17;
                3     3     3     8    10    12     9    11    13     4     7     5     6    14    15    16    17    18    19];
            B= permute(coords, [2, 3, 1]);            
            X = B(:,:,1);
            Y = B(:,:,2);
            Z = B(:,:,3);
            Xlimits = [min(X(:)), max(X(:))];
            Ylimits = [min(Y(:)), max(Y(:))];
            Zlimits = [min(Z(:)), max(Z(:))];
                        
            for s=1:size(X,2)
                S=[X(:,s) Y(:,s) Z(:,s)];                                
                
                % Draw X-Y
                subplot(1,3,1); plot(S(:,1),S(:,2),'r.');
                set(gca,'DataAspectRatio',[1 1 1]);
                axis([Xlimits Ylimits]); title('XY');
                
                for j=1:size(J,2);
                    c1=J(1,j);
                    c2=J(2,j);
                    line([S(c1,1) S(c2,1)], [S(c1,2) S(c2,2)]);
                end
                                
                % Draw ZY
                subplot(1,3,2); plot(S(:,3),S(:,2),'r.');
                set(gca,'DataAspectRatio',[1 1 1]);
                axis([Zlimits Ylimits]); title('ZY');
                
                for j=1:size(J,2);
                    c1=J(1,j);
                    c2=J(2,j);
                    line([S(c1,3) S(c2,3)], [S(c1,2) S(c2,2)]);
                end
                
                % Draw XZY, for best viewpoint, draw XZY instead of XYZ
                subplot(1,3,3);
                plot3(S(:,1),S(:,3),S(:,2),'r.');
                set(gca,'DataAspectRatio',[1 1 1])
                axis([Xlimits, Zlimits, Ylimits])
                
                for j=1:size(J,2);
                    c1=J(1,j);
                    c2=J(2,j);
                    line([S(c1,1) S(c2,1)], [S(c1,3) S(c2,3)], [S(c1,2) S(c2,2)]);
                end
                drawnow;
                pause(1/20)
            end
        end
        
        function create3classTstD()
            classes = 5:7;
            load(sprintf('%s/tstData.mat', MHW_ChaLearnData.dataDir), 'tstD');
            load(sprintf('%s/tstLabel.mat', MHW_ChaLearnData.dataDir), 'tstLabel');
            [newD, newLb] = MHW_ChaLearnData.selData4Classes(tstD, tstLabel, classes, inf); 
            
            idxs = randperm(length(newLb));
            newD = newD(idxs);
            newLb = newLb(idxs);
            ml_save(sprintf('%s/tstData_3classes.mat', MHW_ChaLearnData.dataDir), 'tstD', newD);
            ml_save(sprintf('%s/tstLabel_3classes.mat', MHW_ChaLearnData.dataDir), 'tstLabel', newLb);
        end;

        function create20classTstD()
            load(sprintf('%s/tstData.mat', MHW_ChaLearnData.dataDir), 'tstD');
            load(sprintf('%s/tstLabel.mat', MHW_ChaLearnData.dataDir), 'tstLabel');
            
            idxs = randperm(length(tstLabel));
            tstD = tstD(idxs);
            tstLabel = tstLabel(idxs);
            ml_save(sprintf('%s/tstData_20classes.mat', MHW_ChaLearnData.dataDir), 'tstD', tstD);
            ml_save(sprintf('%s/tstLabel_20classes.mat', MHW_ChaLearnData.dataDir), 'tstLabel', tstLabel);
        end;

        
    end    
end

