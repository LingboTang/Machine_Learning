%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     computeAccuracy_Lambda.m
%     computeAccuracy_TrainSet.m
%     learningAlgorithm.m
%     logisticRegression.m
%     returnBestLambda.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  Load the train and hold our set, as well as the information for performing
%  5 -fold cross validation

load('data.mat')


%% ==================== Part 1: Find best Lambda ====================
%  We start the exercise by creating a function that will use 5-fold
%  cross validation to find the best lambda among the tested values.

fprintf(['Finding the best lambda...\n']);

[bestLambda, acc_vector] = returnBestLambda(X_train, Y_train, cv_ext_indexes);

fprintf(['Best value of lambda found: %f \n(This value shold be 0.30)\n'],bestLambda)
    
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% Find the best model for this set of data using the best
% regularization parameter found.
fprintf(['Using the best lambda to compute the theta vector. \n']);
theta = logisticRegression(X_train,Y_train,bestLambda);

fprintf(['Column 1: Theta obtained. Column 2: expected value of theta.\n']);
expected_vector = [1.93273;0.68816;1.74554;-2.98360;-1.36706;-2.62274;0.27758;-0.47984;...
-0.29140;-0.12714;-2.25122;0.23811;-0.90337;-0.65352;-1.81511;-0.23109;-0.35001;0.21682;...
-0.42029; -0.62009; -0.29278; -1.60037; 0.21460; -0.45834; 0.11202; -0.44075; -0.53619;-1.07783];
comparison = [theta, expected_vector]

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Use the model to compute the accuracy of the model on the hold out set.
predictions = predict(theta,X_hold_out);
accuracy = sum(predictions == Y_hold_out) / length(Y_hold_out);
fprintf(['Accuracy on new, unseen instances: %f \n(This value shold be around 0.782)\n'],accuracy);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Estimate accuracy of the model ====================
%  You will find different estimates of the accuracy of your learning algorithm using
%  different methodologies.

%  Using training set
fprintf(['\nComputing expected accuracy using the train set... \n']);
[trainAcc] = computeAccuracy_TrainSet(X_train,Y_train,cv_ext_indexes);

fprintf(['Accuracy using the training set: %f \n(This value shold be around 0.8526)\n'],trainAcc);
fprintf(['Real accuracy on new instances = 0.782\n']);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% Using CV badly implemented
fprintf(['\nComputing expected accuracy using the CV set (badly implemented) ... \n']);


meanCV_acc = computeAccuracy_Lambda(X_train,Y_train,cv_ext_indexes);
fprintf(['Accuracy using the CV badly implemented set: %f \n(This value shold be around 0.8217)\n'],meanCV_acc);
fprintf(['Real accuracy on new instances = 0.782\n']);

fprintf('\nProgram paused. Press enter to continue.\n');

% Using CV correctly implemented
fprintf(['\nComputing expected accuracy using the CV set (correctly implemented) ... \n']);

[cvAcc] = computeAccuracy_CV(X_train,Y_train,cv_ext_indexes,cv_int_indexes);
fprintf(['Accuracy using the CV correctly implemented set: %f \n(This value shold be around 0.7956)\n'],cvAcc);
fprintf(['Real accuracy on new instances = 0.782\n']);

fprintf('\nProgram paused. Press enter to continue.\n');

