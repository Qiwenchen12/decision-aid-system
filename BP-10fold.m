% 清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

% 读取数据
A = xlsread('data1.xlsx');
rng(0)

% 随机排列索引
temp = randperm(1200);

% 数据打乱
A_shuffled = A(temp, :);

% 定义折数
num_folds = 10;

% 计算每折的大小
fold_size = size(A, 1) / num_folds;

% 初始化数组以存储每一折的准确率和 precision
acc_train_array = zeros(1, num_folds);
acc_test_array = zeros(1, num_folds);
precision_train_array = zeros(num_folds, 5); 
precision_test_array = zeros(num_folds, 5);
recall_train_array = zeros(num_folds, 5);  
recall_test_array = zeros(num_folds, 5);
accuracy_train_class_array = zeros(num_folds, 5);
accuracy_test_class_array = zeros(num_folds, 5);

% 定义超参数搜索空间
param_space = struct('epochs', [500, 1000, 1500], ...
                     'lr', [0.001, 0.01, 0.05, 0.1], ...
                     'num_layers',  [2, 3, 4]);

best_average_accuracy = 0;
best_hyperparameters = [];


% 遍历搜索超参数
for i = 1:1000  % 进行1000次超参数搜索
   % 随机选择迭代次数和学习率
    epochs = param_space.epochs(randi([1, numel(param_space.epochs)]));
    lr = param_space.lr(randi([1, numel(param_space.lr)]));
    
    % 随机选择每层的神经元数量
    num_layers = param_space.num_layers(randi([1, numel(param_space.num_layers)]));
    hidden_neurons = randi([3, 15], 1, num_layers);
    
    % 指定优化器
    optimizer = 'trainrp';  % 使用 trainrp\trainlm
    
    % 设置正则化系数
    regularization_coefficient = 0.2;
    
    % 初始化数组以存储每一折的准确率
    acc_train_array = zeros(1, num_folds);
    acc_test_array = zeros(1, num_folds);
    
    % 遍历 10 折交叉验证
    for fold = 1:num_folds
        % 划分训练集和测试集
        test_indices = round((fold-1)*fold_size) + 1 : round(fold*fold_size);
        train_indices = setdiff(1:size(A, 1), test_indices);

        % 提取训练集和测试集
        P_train = A_shuffled(train_indices, 1:30)';
        T_train = A_shuffled(train_indices, 31)';
        P_test = A_shuffled(test_indices, 1:30)';
        T_test = A_shuffled(test_indices, 31)';

        % 数据预处理 - 归一化
        [P_train, PS] = mapminmax(P_train);
        P_test = mapminmax('apply', P_test, PS);

        % 目标标签进行独热编码
        T_train_encoded = ind2vec(T_train);

        % 定义并训练网络
        net = feedforwardnet(hidden_neurons);
        net.trainParam.showWindow = true;
        net.trainParam.epochs = epochs;
        net.trainParam.lr = lr;
        net.layers{end}.transferFcn = 'softmax';
        for layer = 1:length(net.layers)-1
            net.layers{layer}.transferFcn = 'tansig';  
        end

        % 设置优化器
        net.trainFcn = optimizer;

        % 设置正则化系数
        net.performParam.regularization = regularization_coefficient;

        % 训练网络
        net = train(net, P_train, T_train_encoded);

        % 在训练集上测试
        Y_train = net(P_train);
        [~, Y_train_labels] = max(Y_train);
        train_accuracy = sum(Y_train_labels == T_train) / length(T_train);

        % 在测试集上测试
        Y_test = net(P_test);
        [~, Y_test_labels] = max(Y_test);
        test_accuracy = sum(Y_test_labels == T_test) / length(T_test);
        % 保存准确率
        acc_train_array(fold) = train_accuracy;
        acc_test_array(fold) = test_accuracy;
        % 计算平均训练准确率和测试准确率
        average_train_accuracy = mean(acc_train_array);
        average_test_accuracy = mean(acc_test_array);
        % 遍历每一类别
        for class_label = 1:5
            % 计算混淆矩阵和 precision for 训练集中的每一类
            confusion_matrix_train = confusionmat(T_train, Y_train_labels);
            TP_train_class = confusion_matrix_train(class_label, class_label);
            FP_train_class = sum(confusion_matrix_train(:, class_label)) - TP_train_class;
            FN_train_class = sum(confusion_matrix_train(class_label, :)) - TP_train_class;
            precision_train_class = TP_train_class / (TP_train_class + FP_train_class);
            recall_train_class = TP_train_class / (TP_train_class + FN_train_class);
            accuracy_train_class = sum(Y_train_labels == T_train & Y_train_labels == class_label) / sum(T_train == class_label);


            % 计算混淆矩阵和 precision for 测试集中的每一类
            confusion_matrix_test = confusionmat(T_test, Y_test_labels);
            TP_test_class = confusion_matrix_test(class_label, class_label);
            FP_test_class = sum(confusion_matrix_test(:, class_label)) - TP_test_class;
            FN_test_class = sum(confusion_matrix_test(class_label, :)) - TP_test_class;
            precision_test_class = TP_test_class / (TP_test_class + FP_test_class);
            recall_test_class = TP_test_class / (TP_test_class + FN_test_class);
            accuracy_test_class = sum(Y_test_labels == T_test & Y_test_labels == class_label) / sum(T_test == class_label);


            % 保存每一类的 precision
            precision_train_array(fold, class_label) = precision_train_class;
            recall_train_array(fold, class_label) = recall_train_class;
            precision_test_array(fold, class_label) = precision_test_class;
            recall_test_array(fold, class_label) = recall_test_class;
            accuracy_train_class_array(fold, class_label) = accuracy_train_class;
            accuracy_test_class_array(fold, class_label) = accuracy_test_class;
        end
    end
    precision_train_array = precision_train_array(~any(isnan(precision_train_array), 2), :);
    precision_test_array = precision_test_array(~any(isnan(precision_test_array), 2), :);
    recall_train_array = recall_train_array(~any(isnan(recall_train_array), 2), :);
    recall_test_array = recall_test_array(~any(isnan(recall_test_array), 2), :);
    average_precision_train = mean(precision_train_array, 1);
    average_recall_train = mean(recall_train_array, 1);
    average_precision_test = mean(precision_test_array, 1);
    average_recall_test = mean(recall_test_array, 1);
    average_accuracy_train_class = mean(accuracy_train_class_array, 1);
    average_accuracy_test_class = mean(accuracy_test_class_array, 1);

    fprintf('Iteration %d:\n', i);
    fprintf('Hyperparameters: Epochs=%d, LR=%.4f, Num Layers=%d, Hidden Neurons=%s, Optimizer=%s\n', ...
        epochs, lr, num_layers, mat2str(hidden_neurons), optimizer);
    fprintf('Average Accuracy for Training: %s\n', mat2str(average_accuracy_train_class * 100));
    fprintf('Average Accuracy for Testing: %s\n\n', mat2str(average_accuracy_test_class * 100));
    fprintf('Average Precision for Training: %s\n', mat2str(average_precision_train));
    fprintf('Average Precision for Testing: %s\n\n', mat2str(average_recall_train));
    fprintf('Average Recall for Training: %s\n', mat2str(average_precision_test));
    fprintf('Average Recall for Testing: %s\n\n', mat2str(average_recall_test));


    % 更新最佳超参数
    if average_test_accuracy > best_average_accuracy
        best_average_accuracy = average_test_accuracy;
        best_hyperparameters = struct('epochs', epochs, 'lr', lr, ...
            'num_layers', num_layers, 'hidden_neurons', hidden_neurons, ...
            'optimizer', optimizer);
    end
end
