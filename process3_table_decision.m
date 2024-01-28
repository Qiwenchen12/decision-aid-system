function processedTable = process3_table_decision(inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);

    index_of_3 = Y_table_labels == 3; 
    TABLE3 = inputTable(index_of_3, :);
    processedTable = TABLE3(1, :);
    filename = 'risk level 3.xlsx'; 
    writetable(TABLE3, filename);
end

