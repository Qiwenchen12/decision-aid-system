function processedTable = process4_table_decision(inputTable, net, PS)
    P_table = table2array(inputTable)';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);

    index_of_4 = Y_table_labels == 4; 
    TABLE4 = inputTable(index_of_4, :);
    processedTable = TABLE4(1, :);
    filename = 'risk level 4.xlsx'; 
    writetable(TABLE4, filename);
end

