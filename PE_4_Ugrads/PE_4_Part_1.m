% This file will guide you through the programming assignment # 4.
% You will need to uncomment some of this code as you complete the
% required functions

% Load the file with the required data
load('PE_4_Part_1.mat');


% Create the model using a linear kernel
% ----------------------------------------------------------------------
bestAcc_lin = 0;
paramLin = 'No paramed defined';
[model_linear, paramLin, bestAcc_lin] = findBestModel('linear', X_train, Y_train, folds);

fprintf(['Cross validation accuracy of the best model: ', num2str(bestAcc_lin),' \n (This value should be around 51.4)\n']);
fprintf(['\n The best parameters found were ', paramLin, '. \n This parameters should be -t 0 -c 32768']);
% Create the graphs

figure()
visualizeBoundary(X_train, Y_train, model_linear)
title(['Linear Kernel, Param = ', paramLin]);

fprintf('\n Program paussed, press key to continue \n');
pause;

% Create the model using a polynomial kernel
% ---------------------------------------------------------------------
bestAcc_poly = 0;
paramPoly = 'No paramed defined';
[model_poly, paramPoly, bestAcc_poly] = findBestModel('poly', X_train, Y_train, folds);

fprintf(['Cross validation accuracy of the best model: ',num2str(bestAcc_poly), '\n (This value should be around 59.1)\n']);
fprintf(['\n The best parameters found were ', paramPoly, '. \n This parameters should be -t 1 -c 2048 -d 8']);
% Visualize the boudaries

figure()  
visualizeBoundary(X_train, Y_train, model_poly)
title(['Poly Kernel, Param = ', paramPoly]);

fprintf('\n Program paussed, press key to continue \n');
pause;

% Create the model using a RBF kernel
% ----------------------------------------------------------------------
bestAcc_rbf = 0;
paramRBF = 'No paramed defined';
[model_rbf, paramRBF, bestAcc_rbf] = findBestModel('rbf', X_train, Y_train, folds);

fprintf(['Cross validation accuracy of the best model: ',num2str(bestAcc_rbf),'\n (This value should be around 89.5)\n']);
fprintf(['\n The best parameters found were ', paramRBF, '. \n This parameters should be -t 2 -c 8192 -g 2 \n']);

% Visualize the boudaries
figure() 
visualizeBoundary(X_train, Y_train, model_rbf)
title(['RBF Kernel, Param = ', paramRBF]);
 
