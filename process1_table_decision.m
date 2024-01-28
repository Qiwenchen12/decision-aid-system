function processedTable = process1_table_decision(inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);

    index_of_1 = Y_table_labels == 1; 
    TABLE1 = inputTable(index_of_1, :);
    processedTable = TABLE1(1, :);
    filename = 'risk level 1.xlsx'; 
    writetable(TABLE1, filename);
end