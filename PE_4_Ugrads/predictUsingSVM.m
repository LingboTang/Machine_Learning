% File: predictUsingSVM.m
% -----------------------------------------------------------------------
% This function will train a SVM using fixed parameters. It will then
% use that trained model to make predictions over the data defined in the
% input variable 'dataset'.
function predictions = predictUsingSVM(dataset)
    % ------------------------------------------------------------------
    % ----------------- Modify this part of the code -------------------
    % ------------------------------------------------------------------
    % Replace the values of C and sigma with the parameters that you will
    % use for your submission.
    C = 8192;
    sigma = 2;

    % -----------------------------------------------------------------    
    % Load the data that will be used for training.
    load('PE_4_Part_2.mat');
    
    trainParam = ['-t 2 -c ',num2str(C), ' -g ', num2str(sigma)];
    model = svmtrain(trainLabels, trainSet, trainParam);
    [predictions, ~,~] = svmpredict(zeros(size(dataset,1),1), dataset, model);

end
