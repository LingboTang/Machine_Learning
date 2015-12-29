function [model, bestParam, best_Acc] = findBestModel(kernelType, X_train, Y_train, folds)
% file: findBestModel.m
% ------------------------------------------------------------------------
% This function will find the best set of parameters (among the ones
% tested) given a particular kernel type to use. The output is a SVM model
% created with libsvm.
%
% Inputs:
%
%   - kernelType: String describing the kernel type to use ('linear',
%   'rbf','poly)'
%
%   - X_train: Matrix of n x m, where n is the number of samples in the
%   training set, and m is the number of features.
%
%   - Y_train: Vector of length n, where n is the number of samples. It
%   contains the class of every instance of the training set.
%
%   - folds: A matrix of n x 2, where n is the number of samples. The first
%   column contains the index of every instance (row in X_train). The
%   second column contains the fold to which every point belongs (for
%   5-fold cross-validation). For example, if the first entry of folds is
%   [1,4], it means that the first row of X_train belongs to fold 4.
%
% Outputs:
%   - model is the output of svmtrain (See pdf with instructions for more
%   details).
%
%   - bestParam: string with the parameters used to create the final SVM
%   model. Example: '-t 1 -d 6 -c 8' (See the pdf with the assignments
%   instructions for more details about trainig a SVM).



    % Set the parameters that will be tested inside cross-validation.
    C_vector = [.0312; 0.125; 0.5; 2; 8; 32; 128; 512; 2048; 8192; 32768];
    Sigma_vector = [2^-15; 2^-13; 2^-11; 2^-9; 2^-7; 0.0312; 0.125; .5; 2; 8];
    Degree_vector = [2;3;4;5;6;7;8];
    
    % Determine the number of folds.
    K = length(unique(folds(:,2)));
    bestParam = '';
    best_Acc = 0;

    switch lower(kernelType)
        case 'linear'
            % Use 5-fold cross validation to select the best parameters.
            % ----------------- Your code here--------------------------
            for i = 1:length(C_vector)
                bestThisAcc = 0;
                for j = 1:K
                    [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X_train, Y_train,folds, j);
                    bestCParam = C_vector(i);
                    bestThisParam = ['-t 0 -c ',num2str(bestCParam), ' -q'];
                    model_opt = svmtrain(labelsTrain, trainSet, bestThisParam);
                    [predictions, accuracy, ~] = svmpredict(labelsTest, testSet, model_opt);
                    bestThisAcc = bestThisAcc + accuracy(1);
                end
                if bestThisAcc > best_Acc
                    best_Acc = bestThisAcc;
                    bestCParam = C_vector(i);
                    bestThisParam = ['-t 0 -c ',num2str(bestCParam), ' -q'];
                    bestParam = bestThisParam;
                end
            end
            best_Acc = best_Acc/K;
            % Now use the best parameters that you found, and the entire
            % training set to output your final model
            model = svmtrain(Y_train, X_train, bestParam);
        
        case 'poly'
            % Use 5-fold cross validation to select the best parameters.
            % ----------------- Your code here--------------------------
            combination = combvec(C_vector', Degree_vector');  
            for i = 1:length(combination)
                bestThisAcc = 0;
                for j = 1:K
                    [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X_train, Y_train,folds, j);
                    bestC = combination(1, i);
                    bestD = combination(2, i);
                    bestThisParam = ['-t 1 -c ',num2str(bestC), ' -d ', num2str(bestD) ' -q'];
                    model_opt = svmtrain(labelsTrain, trainSet, bestThisParam);
                    [predictions, accuracy, ~] = svmpredict(labelsTest, testSet, model_opt);
                    bestThisAcc = bestThisAcc + accuracy(1);
                end
                if bestThisAcc > best_Acc
                    best_Acc = bestThisAcc;
                    bestC = combination(1, i);
                    bestD = combination(2, i);
                    bestThisParam = ['-t 1 -c ',num2str(bestC), ' -d ', num2str(bestD) ' -q'];
                    bestParam = bestThisParam;
                end
            end
            best_Acc = best_Acc/K;

            % Now use the best parameters that you found, and the entire
            % training set to output your final model
            model = svmtrain(Y_train, X_train, bestParam);
        
        
        case 'rbf'
            % Use 5-fold cross validation to select the best parameters.
            % ----------------- Your code here--------------------------
            combination = combvec(C_vector', Sigma_vector');  
            for i = 1:length(combination)
                bestThisAcc = 0;
                for j = 1:K
                    [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X_train, Y_train,folds, j);
                    bestC = combination(1, i);
                    bestS = combination(2, i);
                    bestThisParam = ['-t 2 -c ',num2str(bestC), ' -g ', num2str(bestS) ' -q'];
                    model_opt = svmtrain(labelsTrain, trainSet, bestThisParam);
                    [predictions, accuracy, ~] = svmpredict(labelsTest, testSet, model_opt);
                    bestThisAcc = bestThisAcc + accuracy(1);
                end
                if bestThisAcc > best_Acc
                    best_Acc = bestThisAcc;
                    bestC = combination(1, i);
                    bestS = combination(2, i);
                    bestThisParam = ['-t 2 -c ',num2str(bestC), ' -g ', num2str(bestS) ' -q'];
                    bestParam = bestThisParam;
                end
            end
            best_Acc = best_Acc/K;
            % Now use the best parameters that you found, and the entire
            % training set to output your final model
            model = svmtrain(Y_train, X_train, bestParam);
    end
end
