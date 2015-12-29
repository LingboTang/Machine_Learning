%% learnDecisionTree
% The algorithm for this code was outlined in the lecture notes:
%       https://eclass.srv.ualberta.ca/pluginfile.php/2312202/mod_resource/content/4/6a-DecisionTree.pdf
% on slide labeled with the number 27.
% 
% Inputs: 
%       examples            - set of examples [X1, ..., Xn,Class]
%       attribute           - attribute descriptions
%                             an [n x 1] vector of structs with fields: 
%                                   'id'    - a unique id number
%                                   'name'  - human understandable name of attribute
%                                   'value' - possible attribute values
%
%                             Example: attribute(1) = 
%                                      id: 1
%                                      name: 'Clump Thickness'
%                                      value: [1 2 3 4 5 6 7 8 9 10]
% 
% Outputs:
%       tree                - Decision Tree


function tree = learnDecisionTree(examples, attribute, default)

    %% Here's a helpful structure for creating a tree in MATLAB
    %  Each node in the tree is struct with five fields. 
    %         'attribute'   - integer id of the attribute we split on
    %         'isleaf'      - is 'true' if the node is a leaf 
    %                         and 'false' otherwise
    %         'class'       - is 'null' if the node is not a leaf. 
    %                         If node is a leaf, class= '0' or '1'
    %         'children'    - Is 'null' if the node is a leaf. 
    %                         Otherwise, it is a cell {} where 
    %                         tree.children{i} is  the subtree when the 
    %                         tree.attribute takes on value tree.value(i).
    %                         Is 'null' if the node is a leaf.
    %         'value'       - a vector of values that the attribute can
    %                         take. Is 'null' if the node is a leaf.
    %         'num_1'       - The number of training examples in class = 1
    %                         at the node.
    %         'num_0'       - The number of training examples in class = 0
    %                         at the node.
    %         'num_tot'     - The total number of training examples at the
    %                         node.
    %
    %  Example (non-leaf node):
    % 
    %     attribute: [1x1 struct]
    %        isleaf: 0
    %         class: 'null'
    %      children: {1x10 cell}
    %         value: [1 2 3 4 5 6 7 8 9 10]
    %         num_1: 43
    %         num_0: 2
    %       num_tot: 45
    %         
    %  
    %  Example (leaf node):
    %  
    %     attribute: 'null'
    %        isleaf: 1
    %         class: '0'
    %      children: 'null'
    %         value: 'null'
    %         num_1: 43
    %         num_0: 2
    %       num_tot: 45
    
    tree = struct('attribute','null',...
                  'isleaf','null',...
                  'class',default,...
                  'children','null',...
                  'value','null',...
                  'num_1',-1,...
                  'num_0',-1,...
                  'num_tot',-1);             

    % If there are no examples to classify, return
    num_examples = size(examples(:,end),1);
    
    if num_examples == 0
        return
    end
    %% 1.) If all examples have the same classification, create a 
    %      tree leaf node with that classification and return
    
    % Get all the non-repeated values from the example
    unique_examples = unique(examples(:,end));
    
    if (length(unique_examples) == 1),
        % child
        tree.isleaf = 1;
        % classifier is the unique_examples
        tree.class = unique_examples;
        % Just use the portion of the num_examples
        tree.num_0 = (tree.class==0)*num_examples;
        % that has the classifier
        tree.num_1 = (tree.class==1)*num_examples;
        % num_tot is the total number of examples
        tree.num_tot = num_examples;
        return
    end
        
    %% 2.) If attributes is empty, create a leaf node with the
    %      majority classification and return.
    
    % Is empty
    if length(attribute') == 0
        % is child
        tree.isleaf = 1;
        % The majority classification is the new default
        % which means true positive label should be more than half
        
        % Only positive is one, so we don't need loop to count
        num_ones = sum(examples(:,end));
        total = length(examples);
        % So does zeors
        num_zeros = total-num_ones;
        tree.class = num_ones>total/2;
        tree.num_0 = num_zeros;
        tree.num_1 = num_ones;
        tree.num_tot = total;
        return
    end
       
    %% 3.) Find the best attribute -- the attribute with the lowest uncertainty    %
    
    % Initialize the bestattributes
    
    bestlabels = examples(1,:);
    reallen = size(bestlabels)-1;
    bestattributes = zeros(reallen);
    
    % Iterate
    for j=1:size(bestattributes',1)
        thisthislabel = examples(:,j);
        % Need to find the unique_labels first
        unique_this = unique(thisthislabel);
        % The best attributes must have minimized uncertainty!
        bestattributes(j) = uncert(j, unique_this, examples);
    end
    [~, best_i] = max(bestattributes);
    
    %% 4.) Make a non-leaf tree node with root 'best'
    
    % Best attributes is the attribute with argbest(attributes) as index
    tree.attribute = attribute(best_i);
    % Make it a root
    tree.isleaf = 0;
    tree.value = tree.attribute.value;
    num_of_posi = sum(examples(:,end));
    total = length(examples(:,end));
    num_of_neutual = total - num_of_posi;
    tree.num_0 = num_of_neutual;
    tree.num_1 = num_of_posi;
    tree.num_tot = total;
    
    
    %% 5.) For each value v_i that the best attribute can take, do the following:
    %     a.) examples_i <-- elements of examples where the best attribute has value v_i
    %     b.) subtree <-- recursive call to learnDecisionTree with inputs:
    %              examples_i
    %              all attributes but the best
    %              the majority value of the examples
    %     c.) add branch to tree with label vi and subtree
    my_value_list = tree.attribute.value;
    % Make each of tree children to be the new struct
    tree.children = cell(length(my_value_list),1);
    for j=1:length(tree.attribute.value')
        % find the best examples of that matched to the value of our
        % attribue
        value_slots = find(examples(:,best_i)==tree.attribute.value(j));
        examples_i = examples(value_slots,:);
        % initialize the new value
        examples_i(:,best_i) = [];
        % initialize the new attribute
        attribute_i = attribute;
        attribute_i(:,best_i) = [];
        % add the newly defined back to the tree child
        tree.children{j} = learnDecisionTree(examples_i, attribute_i, default);
    end
    return
end

%% You may wish to write a function that...
%  Computes the uncertainty of the i-th attribute when given:
%        i                - the id of the attribute
%        attribute_vals   - the vector of possible values that attribute
%                           can take
%        examples         - the set of examples on which you'll compute
%                           the information gain
%% 
function value = uncert(i, attribute_vals, examples)
    classes = examples(:,end);
    value = 0;
    for j=1:length(attribute_vals)
       % Find all the slots index
       slots = find(examples(:,i)==attribute_vals(j));
       % portion is the sum of all the slots
       prob = sum(classes(slots));
       cases = length(classes(slots))-prob;
       thisportion = length(slots)/length(classes);
       value =value + entropy(prob, cases)*thisportion;
    end
    % Have to get the abs number of entropy
    value = -value;
end


%% You may wish to have an entropy function that...
%  Computes entropy when given:
%        p               - the number of class = 1 examples
%        n               - the number of class = 0 examples
%%
function en = entropy(p,n)
    % pi = (|{(x,y) belongs to S | y =i}|)/|S|
    p1 = p/(p+n);
    p2 = n/(p+n);
    % Entropy(S) = -sum(from 1 to c) pi*log2(pi)
    en = -(p1.*log2(p1)+p2.*log2(p2));
end