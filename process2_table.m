function processedTable = process2_table( inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);

    index_of_3 = Y_table_labels == 3; % 将index_of改为index_of_4
    inputTable = inputTable(index_of_3, :); 

    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);

    [~, sorted_indices] = sort(Y_table(2, :), 'descend');
    top_10_percent = ceil(0.1 * numel(sorted_indices));
    selected_indices = sorted_indices(1:top_10_percent);
    processedTable = inputTable(selected_indices, :);
end
