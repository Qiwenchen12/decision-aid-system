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

% 测试网络
Y_test = net(P_test);
[~, Y_test_labels] = max(Y_test); % 将独热编码的输出转换为整数标签

% 计算分类准确率
acc_test = sum(Y_test_labels == T_test) / N;

% 计算训练集上的准确率和MSE
Y_train = net(P_train);
[~, Y_train_labels] = max(Y_train); % 将独热编码的输出转换为整数标签
acc_train = sum(Y_train_labels == T_train) / M;


num_features = 30; % 特征数量
feature_names = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', ...
    'A11', 'A12', 'A13', 'A14', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4', 'C5', ...
    'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'};
num_permutations = 1000; % 每个特征的打乱次数

feature_importance = zeros(num_features, num_permutations); % 初始化特征重要性矩阵

for feature = 1:num_features
    for permutation = 1:num_permutations
        permuted_P_test = P_test; % 创建一个用于置换特征值的测试集副本
        permuted_P_test(feature, :) = randperm(N); % 打乱当前特征的值

        permuted_Y_test = net(permuted_P_test); % 使用打乱特征值后的测试集进行预测
        [~, permuted_Y_test_labels] = max(permuted_Y_test); % 将独热编码的输出转换为整数标签

        permuted_acc_test = sum(permuted_Y_test_labels == T_test) / N; % 计算打乱特征后的分类准确率

        feature_importance(feature, permutation) = acc_test - permuted_acc_test; % 计算特征的重要性（性能差异）
    end
end

% 绘制特征重要性散点图
figure;
hold on;

for feature = 1:num_features
    x = repmat(feature, 1, num_permutations); % 横轴特征序号
    y = feature_importance(feature, :); % 纵轴特征重要性值
    
    % 点的颜色深浅与特征重要性值相关
    color = y;
    
    % 调整点的间距
    pointJitter = 0.2; % 点的微小偏移
    x = x + (rand(size(x)) - 0.5) * pointJitter;
    
    % 设置点的大小和透明度
    pointSize = 50;
    pointAlpha = 0.8;
    
    scatter(x, y, pointSize, color, 'filled', 'MarkerFaceAlpha', pointAlpha);
end

hold off;

% 调整坐标轴范围
xlim([0.5 num_features+0.5]); % 横轴范围
ylim([min(feature_importance(:))-0.05 max(feature_importance(:))+0.05]); % 纵轴范围

% 设置坐标轴刻度和标签
xticks(1:num_features); % 横轴刻度位置
xticklabels(1:num_features); % 横轴刻度标签
xticklabels(feature_names);
xlabel('Indicator');
ylabel('Indicator Importance');
colormap('jet');
colorbar;


% 绘制特征重要性的箱型图
figure;
boxplot(feature_importance', 'Labels', 1:num_features);
xlabel('Indicator');
xticklabels(feature_names);
ylabel('Indicator Importance');

median_values = median(feature_importance, 2);
quartile_values = prctile(feature_importance, [25, 75], 2);

% Compute the interquartile range (IQR)
iqr_values = quartile_values(:,2) - quartile_values(:,1);

% Create a table with the results
factor_names = cellstr(num2str((1:num_features)', 'Feature %d'));
table_data = table(factor_names, median_values, quartile_values(:,1), quartile_values(:,2), iqr_values, ...
                   'VariableNames', {'Factor', 'Median', 'Q1', 'Q3', 'IQR'});

% Display the table
disp(table_data);




median_values = median(feature_importance, 2);
quartile_values = prctile(feature_importance, [25, 75], 2);

% Compute the interquartile range (IQR)
iqr_values = quartile_values(:,2) - quartile_values(:,1);

% Create a table with the results
factor_names = cellstr(num2str((1:num_features)', 'Feature %d'));
table_data = table(factor_names, median_values, quartile_values(:,1), quartile_values(:,2), iqr_values, ...
                   'VariableNames', {'Factor', 'Median', 'Q1', 'Q3', 'IQR'});

% Display the unsorted table
disp("Unsorted Table:");
disp(table_data);

% Sort the table based on the median values in ascending order
sorted_table = sortrows(table_data, 'Median');

% Display the sorted table
disp("Sorted Table:");
disp(sorted_table);

