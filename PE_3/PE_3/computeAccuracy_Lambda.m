%% File: computeAccuracy_Lambda
% ------------------------------------------------------------------------
% First, use the function returnBestLambda to decide the best parameter
% for Lambda (Remember that this function internally performs 5-fold CV). 
% Then use 5-fold Cross Validation (with the best lambda) and the functions
% logisticRegression and predict to estimate the accuracy of your
% classifier.

function meanCV_acc = computeAccuracy_Lambda(Xtrain,Ytrain,folds)

    K = 5;
    cv_acc = zeros(K,1);
    
    % -------------------- Your code here -----------------------------
    % Find the best value of lambda using the training set and the function
    % findBestLambda.m


    % Now perform cross validation on the trainSet. Use the function 
    % logisticRegression and the best value of lambda found in the previous
    % part of the exercise inside every fold. Store the accuracy of the 
    % fold i in cv_acc(i).

    [bestlambda,~] = returnBestLambda(Xtrain,Ytrain,folds);
    
    for i=1:K
        [trainSet, testSet, labelsTrain, labelsTest] = generateSets(Xtrain,Ytrain,folds,i);
        Theta_opt = logisticRegression(trainSet, labelsTrain, bestlambda);
        val = predict(Theta_opt, testSet);
        counter = 0;
        for a=1:length(val)
            if val(a) == labelsTest(a)
                counter = counter+1;
            end
        end
        cv_acc(i) = cv_acc(i) + counter/length(val);
    end
    

    % -----------------------------------------------------------------

    meanCV_acc = mean(cv_acc);
    
end
