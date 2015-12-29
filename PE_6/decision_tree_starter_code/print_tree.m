function print_tree(tree)
    print_tree_r(tree,'root','');
end

function print_tree_r(tree,parent,prefix)

    if tree.isleaf,
        fprintf('Class : %d   +/- = [%d , %d] \n', tree.class, tree.num_1, tree.num_0)
    else
       prefix = [prefix , ' |'];
        for i = 1:length(tree.children)
            fprintf('%s-Attribute : %s = %d ', prefix, tree.attribute.name, tree.value(i))
            if ~tree.children{i}.isleaf,
                fprintf('\n');
            end
            print_tree_r(tree.children{i}, tree , prefix);
        end
    end

end