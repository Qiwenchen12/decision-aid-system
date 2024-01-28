function processedTable = process4_table( inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);

    [~, sorted_indices] = sort(Y_table(4, :), 'descend');
    top_10_percent = ceil(0.1 * numel(sorted_indices));
    selected_indices = sorted_indices(1:top_10_percent);
    processedTable = inputTable(selected_indices, :);
end
