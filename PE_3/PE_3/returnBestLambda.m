%% File: returnBestLambda.m
% ------------------------------------------------------------------------
% This function will choose the best regularization parameter to use
% with logistic regression. It will determine that value using K-fold
% cross validation.
% 
% Inputs:
%   - X: matrix of n x m, where n is the number of samples, and m is the
%        number of features.
%   - y: column vector of length n that contains the labels.
%   - folds: matrix of p x 2, where p is the number of samples
%        in the training set X. The first column contains the sample index
%        (row) of the matix X, the second column contains the fold to which
%        it belongs. For example:
%      
%        folds_class_1(1,:) = [20,4]
%        means that the row 20 of X belongs to the fold # 4.

function [bestLambda, accuracy_vector] = returnBestLambda(X,y,folds)
    % Number of K for K-fold cross validation.
    K = 5;
    
    % Define the values of lambda to try
    lambda_vector = [0,0.01, 0.03, 0.1, 0.3, 1, 3,10,30,100];
    
    % Initialize accuracy_vector. accuracy_vector(i) is the cross
    % validation accuracy (average over the 5 folds) using lambda_vector(i)
    % as regularization parameter. For example: accuracy_vector(1) is the
    % cross-validation accuracy using lambda = 0.01
    accuracy_vector = zeros(length(lambda_vector),1);
    
    % Iterate through all the lambdas and perform 5 fold cross validation using
    % every value of lambda.
    %
    % Hint: For every lambda, inside every iteration in 5-fold cross validation, you can create
    %       a temporal trainSet and testSet, and then use the functions 
    %       logistic regression and predict. Finally you can compute the accuracy
    %       which is the proportion of labels correctly classofied in the temporal
    %       test Set that you created. Finally, average the accuracy over the 5 folds
    %       to compute the cross validation accuracy associated with that particuar value
    %       of lambda. You can create the sets automatically using the function: generateSets.
    %
    %       For example, if you want to create the temporal sets for the fold # 3, you will
    %       call the function in the following way:
    %
    %    [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X,y,folds, 3);

    % ----------------------- Your code here ----------------------------------------

    
    for i=1:length(lambda_vector)
        for j =1:K
            [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X,y,folds,j);
            Theta_opt = logisticRegression(trainSet, labelsTrain, lambda_vector(i));
            val = predict(Theta_opt, testSet);
            accuracy_vector(i) = accuracy_vector(i) + sum(val==labelsTest)/length(val);
        end
        accuracy_vector(i) = accuracy_vector(i)/K;
    end
    
    % -------------------------------------------------------------------------------
    
    % Determine the best value of lambda.
    [~, indexMax] = max(accuracy_vector);
    bestLambda = lambda_vector(indexMax);
end
