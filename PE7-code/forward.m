function [alpha, P] = forward(O, phi, A, B)
% Given size-T observation, O, and HMM parameters corresponding to k states,
% forward.m computes and returns forward matrix, alpha, of size T*k, where
% alpha(t,i) = Prob(O_{1},O_{2},....O_{t}, D_t=i) and P, of size 1*T, where
% P(t)=Prob(O_{1},O_{2},....O_{t})
%
% Input:
%   O: sequence of Observations (1*T)
%   phi: initial state distribution of HMM (1*k)
%   A: HMM transition matrix (k*k)
%   B: HMM emission matrix (m*k)
%
% Returns:
%   alpha: forward matrix (T*k)
%   P: probability of the observation sequence (1*T)
%
%   See Eqn 18 in Rabiner 1989 for details
T = length(O); % size of observation sequence
m= size(B,1);  % number of possible observed values
k = size(A,1);  % number of possible states
alpha = zeros(T, k);


%Your code goes here

alpha(1,:) = phi.*B(O(1),:); 
for i=2:T
  previous_State = alpha(i-1,:);
  present_Emission = B(O(i),:);
  previous_Transmission = previous_State*A;
  alpha(i,:) = previous_Transmission.*present_Emission;
end

P = sum(alpha');

end
