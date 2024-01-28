function processedTable = process2_table_decision(inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);

    index_of_2 = Y_table_labels == 2; 
    TABLE2 = inputTable(index_of_2, :);
    processedTable = TABLE2(1, :);
    filename = 'risk level 2.xlsx'; 
    writetable(TABLE2, filename);
end
