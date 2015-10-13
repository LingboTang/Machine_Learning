%% File: computeAccuracy_CV
% ------------------------------------------------------------------------
% You will implement 5-fold cross validation (external cross validation)
% , and inside every fold you will call the function 'learningAlgorithm',
% which uses also cross validation to find the best parameters of lambda
% (internal cross validation).
%
% The information for the external cross validation is defined in the 
% matrix 'folds', which is a matrix of m x2, where m is the number of samples
% in Xtrain.
%
% In the external cross validation set, you will create 5 temporal training
% sets and test sets (one pair for every fold). You can create these models
% using the function;
%
% [tempTrainSet, tempTestSet, labelsTrain, labelsTest] = generateSets(Xtrain, Ytrain, folds, k)
%
% where k is the current fold (1,2,3,4 or 5).
% 
% Then, you will build a model using one of these training sets. In order 
% to select the best lambda for a particular training set, you might use 
% the function learningAlgorithm, which in turn performs 5-fold cross validation. 
% The information for this internal cross validation is contained in the cell 
% 'subfolds'. For example if you want to construct the model for the second 
% iteration of the external cross validation (k = 2), you can use the function as:
%
%  theta = learningAlgorithm(tempTrainSet, labelsTempTrainSet, subfolds{2})
%
% Finally, you will compute the accuracy of this model on the temporal test set.
% (For this example, where k = 2, you would store that accuracy in cv_acc(2)
 

function [cvAcc] = computeAccuracy_CV(Xtrain,Ytrain,folds,subfolds)

    K = 5;
    cv_acc = zeros(K,1);
    % --------------------------- Your code here -----------------------------

    for i=1:K
        [tempTrainSet, tempTestSet, labelsTrain, labelsTest] = generateSets(Xtrain, Ytrain, folds, i);
        theta = learningAlgorithm(tempTrainSet, labelsTrain, subfolds{i});
        val = predict(theta,tempTestSet);
        cv_acc(i) = sum(val == labelsTest)/length(val);
    end
        

    % ------------------------------------------------------------------------


    cvAcc = mean(cv_acc);
    
end