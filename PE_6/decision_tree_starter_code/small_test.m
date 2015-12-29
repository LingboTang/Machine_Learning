load planets
tree = learnDecisionTree(train_set, attribute, 0);
print_tree(tree);
classification = classify(tree, train_set(1,:))
%load breast_cancer_dataset
%tree = learnDecisionTree(train_set, attribute, 0);
%print_tree(tree)
%classification = classify(tree, train_set(1,:))
%% If your learnDecisionTree() and classify() functions work,
%  you should see the following output:
%
%  |-Attribute : Size = 0 
%  | |-Attribute : Orbit = 0 Class : 1   +/- = [127 , 11] 
%  | |-Attribute : Orbit = 1 Class : 0   +/- = [43 , 238] 
%  |-Attribute : Size = 1 
%  | |-Attribute : Orbit = 0 Class : 0   +/- = [16 , 123] 
%  | |-Attribute : Orbit = 1 Class : 1   +/- = [163 , 29] 
% 
% classification =
% 
%      0
