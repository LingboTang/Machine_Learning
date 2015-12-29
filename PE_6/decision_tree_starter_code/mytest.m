load breast_cancer_dataset
tree = learnDecisionTree(train_set, attribute, 0);
print_tree(tree)
classification = classify(tree, train_set(1,:))