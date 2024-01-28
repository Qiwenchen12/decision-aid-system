%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

% 读取数据
A = xlsread('data.xlsx');
rng(1)

% 训练集大小
train_size = 1200;
% 提取训练集
P_train = A(1:train_size, 1:30)';
T_train = A(1:train_size, 31)';
% 计算训练样本数和特征数
M = size(P_train, 2);

% 数据预处理 - 归一化
[P_train, PS] = mapminmax(P_train);
% 使用同样的缩放参数对测试集进行归一化
P_test = mapminmax('apply', A(train_size+1:end, 1:30)', PS);
% 提取测试集
T_test = A(train_size+1:end, 31)';

% 计算测试样本数
N = size(P_test, 2);

% 目标标签进行独热编码
T_train_encoded = ind2vec(T_train); % 使用MATLAB的ind2vec函数进行独热编码

% 定义网络结构
hiddenLayerSizes = [10 5];
net = feedforwardnet(hiddenLayerSizes);
for i = 1:length(hiddenLayerSizes)
    net.layers{i}.transferFcn = 'tansig';
end

% 设置训练参数
net.trainParam.showWindow = true;
net.trainParam.epochs = 1500;
net.trainParam.lr = 0.1;
net.performParam.regularization = 0.2;
net.trainFcn = 'trainrp';
% 训练网络
net = train(net, P_train, T_train_encoded); % 使用独热编码后的目标标签进行训练
net.layers{end}.transferFcn = 'softmax'; % 设置输出层的激活函数为Softmax



run r0.m
P_table0 = table2array(table0)';
P_table0_normalized = mapminmax('apply', P_table0, PS);
Y_table0 = net(P_table0_normalized);
[~, Y_table0_labels] = max(Y_table0);



i = 1;
while true
    % Formulate the filename for the current ri.m script
    ri_script_name = sprintf('r%d.m', i);
    tablei_var_name = sprintf('table%d', i);
    run(ri_script_name);
    
    P_table = table2array(eval(tablei_var_name))';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);
    
    % Check if Y_table_labels contains 4
    if any(Y_table_labels == 4)
        fprintf('%s contains the value 4.\n', tablei_var_name);
        break; % Exit the loop if 4 is found
    else
        fprintf('%s does not contain the value 4.\n', tablei_var_name);
    end
    
    eval(sprintf('%s = process4_table(%s, net, PS);', tablei_var_name, tablei_var_name));
    
    % Increment i for the next iteration
    i = i + 1;
end

run r2.m
table2 = process4_table_decision(table2, net, PS);

i = 3;
while true
    % Formulate the filename for the current ri.m script
    ri_script_name = sprintf('r%d.m', i);
    tablei_var_name = sprintf('table%d', i);
    run(ri_script_name);
    
    P_table = table2array(eval(tablei_var_name))';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);
    
    % Check if Y_table_labels contains 4
    if any(Y_table_labels == 3)
        fprintf('%s contains the value 3.\n', tablei_var_name);
        break; % Exit the loop if 4 is found
    else
        fprintf('%s does not contain the value 3.\n', tablei_var_name);
    end
    
    eval(sprintf('%s = process3_table(%s, net, PS);', tablei_var_name, tablei_var_name));
    
    % Increment i for the next iteration
    i = i + 1;
end

run r10.m
table10 = process3_table_decision(table10, net, PS);

i = 11;
while true
    % Formulate the filename for the current ri.m script
    ri_script_name = sprintf('r%d.m', i);
    tablei_var_name = sprintf('table%d', i);
    run(ri_script_name);
    
    P_table = table2array(eval(tablei_var_name))';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);
    
    % Check if Y_table_labels contains 4
    if any(Y_table_labels == 2)
        fprintf('%s contains the value 2.\n', tablei_var_name);
        break; % Exit the loop if 4 is found
    else
        fprintf('%s does not contain the value 2.\n', tablei_var_name);
    end
    
    eval(sprintf('%s = process2_table(%s, net, PS);', tablei_var_name, tablei_var_name));
    
    % Increment i for the next iteration
    i = i + 1;
end

run r19.m
table19 = process2_table_decision(table19, net, PS);


i = 20;
while true
    % Formulate the filename for the current ri.m script
    ri_script_name = sprintf('r%d.m', i);
    tablei_var_name = sprintf('table%d', i);
    run(ri_script_name);
    
    P_table = table2array(eval(tablei_var_name))';
    P_table_normalized = mapminmax('apply', P_table, PS);
    Y_table = net(P_table_normalized);
    [~, Y_table_labels] = max(Y_table);
    
    % Check if Y_table_labels contains 4
    if any(Y_table_labels == 1)
        fprintf('%s contains the value 1.\n', tablei_var_name);
        break; % Exit the loop if 4 is found
    else
        fprintf('%s does not contain the value 1.\n', tablei_var_name);
    end
    
    eval(sprintf('%s = process1_table(%s, net, PS);', tablei_var_name, tablei_var_name));
    
    % Increment i for the next iteration
    i = i + 1;
end

run r41.m
table41 = process1_table_decision(table41, net, PS);