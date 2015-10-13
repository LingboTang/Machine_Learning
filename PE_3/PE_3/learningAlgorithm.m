function [theta] = learningAlgorithm(X,y, folds)
    theta = 0;

    % Find the best regularization parameter for this dataset.
    % -------- Your code here ----------
    [bestLambda, accuracy_vec] = returnBestLambda(X,y,folds);


    % Find the best model for this set of data using the best
    % regularization parameter found.
    % -------- Your code here ----------
    theta = logisticRegression(X, y, bestLambda);

	

end
