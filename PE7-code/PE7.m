function [phi,alpha,P_O, beta, dstar,pstar,gamma,prob_watch, prob_no_watch] = PE7()

%% CMPUT 466/551 (2015)
%% PE#7 script

%% HMM State transition matrix
A = [0.80, 0.20; 0.1 0.9];

%% HMM Emission Matrix
B = [1/6 4/5; 1/6 1/25; 1/6 1/25; 1/6 1/25; 1/6 1/25; 1/6 1/25];

%% Observations from HMM
O = [4, 1, 2, 3, 1, 3, 1, 1, 5, 6];

%% initial state distribution
%This is P(D0)
phi_0 = [0.5 0.5];

%Using the initial state distribution, predict the state distribution
%before evidence i.e. P(D1)
%You code here for (a)

fprintf('(a) P(D_1) \n')

% Modify this variable as needed.
phi= phi_0 * A;
a = phi


%% Uncomment this portion to use appropriate variables  
alpha = 0;
P_O = 0;
[alpha, P_O] = forward(O, phi, A, B);

beta = 0;
beta = backward(O, A, B);

dstar = 0;
pstar = 0;
[dstar,pstar] = viterbi(O, phi, A, B);
pstar1 = pstar;
%% Now answer the Questions:

% (b): P(D_t = r | S_1:t)
fprintf('(b): P(D_t = r | S_1:t) \n')
% alpha is needed here
b = alpha(:,2)

% (c): P(D_t = r | S_1:10)
fprintf('(c): P(D_t = r | S_1:10) \n')
% gamma is needed here (See Eqn 27 in Rabiner 1989 for details)
gamma = 0;
gamma = (alpha(:,2).*beta(:,2))/(sum(alpha(:,2).*beta(:,2)))
c = gamma

% (d): argmax_d P(D=d | S_1:10)
fprintf('(d): argmax_d P(D=d | S_1:10) \n')
% use viterbi algorithm
d = dstar

% (e): P_+L(S_1:10) and P_-L(S_1:10)
fprintf('(e): P_+L(S_1:10) and P_-L(S_1:10) \n')
A_L = [0.75 0.25; 0.05 0.95];

[alpha, P_O] = forward(O, phi, A_L, B);
beta = backward(O, A_L, B);
[dstar,pstar2] = viterbi(O, phi, A_L, B);
b = sum(alpha(:,2));
c = sum(beta(:,2));
d = dstar;


% Modify the values of these variables as needed.
prob_watch = pstar2
prob_no_watch = pstar1

end