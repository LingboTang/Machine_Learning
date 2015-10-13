%% File: computeAccuracy_TrainSet
% ------------------------------------------------------------------------
% First, use the function learningAlgorithm to get a classifier. Then
% compute the accuracy of your classifier using the X_train.
% 
% After that, compute the accuracy of the SAME classifier over the the 
% set X_hold_out. (You must not train a new classifier here, use the same
% values of lambda that you used for the first part).

function [trainAcc] = computeAccuracy_TrainSet(Xtrain,Ytrain,folds)

    trainAcc = 0;
    % Use the functions findBestLambda, and logistic regression to create a 
    % a classifier based on the training set. (Alternatively, you can use the 
    % function learningAlgorithm that you computed previously).
    % 
    % Then, compute the accuracy of that classifier using the training set.

    % ------------------------- Your code here ------------------------------

    [theta] = learningAlgorithm(Xtrain,Ytrain, folds);
    val = predict(theta, Xtrain);
    trainAcc = sum(val == Ytrain)/length(val);


    % -----------------------------------------------------------------------


    
end
