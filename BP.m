% 清空环境变量
warning off
close all
clear
clc

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
T_train_encoded = ind2vec(T_train);

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
net = train(net, P_train, T_train_encoded);
net.layers{end}.transferFcn = 'softmax';

% 测试网络
Y_test = net(P_test);
[~, Y_test_labels] = max(Y_test);

% 计算测试集整体准确率
acc_test = sum(Y_test_labels == T_test) / N;

% 计算测试集每一类的准确率
class_labels_test = unique(T_test);
num_classes_test = length(class_labels_test);
class_accuracy_test = zeros(1, num_classes_test);

for i = 1:num_classes_test
    class_idx_test = (T_test == class_labels_test(i));
    correct_classifications_test = sum(Y_test_labels(class_idx_test) == class_labels_test(i));
    total_samples_in_class_test = sum(class_idx_test);
    class_accuracy_test(i) = correct_classifications_test / total_samples_in_class_test;
end

% 输出测试集结果
for i = 1:num_classes_test
    fprintf('Class %d accuracy on testing set: %.2f%%\n', class_labels_test(i), class_accuracy_test(i) * 100);
end

% 训练集的准确率和每一类的准确率
Y_train = net(P_train);
[~, Y_train_labels] = max(Y_train);
acc_train = sum(Y_train_labels == T_train) / M;

% 计算训练集每一类的准确率
class_labels_train = unique(T_train);
num_classes_train = length(class_labels_train);
class_accuracy_train = zeros(1, num_classes_train);

for i = 1:num_classes_train
    class_idx_train = (T_train == class_labels_train(i));
    correct_classifications_train = sum(Y_train_labels(class_idx_train) == class_labels_train(i));
    total_samples_in_class_train = sum(class_idx_train);
    class_accuracy_train(i) = correct_classifications_train / total_samples_in_class_train;
end

% 输出训练集结果
for i = 1:num_classes_train
    fprintf('Class %d accuracy on training set: %.2f%%\n', class_labels_train(i), class_accuracy_train(i) * 100);
end



% 计算混淆矩阵
C_test = confusionmat(T_test, Y_test_labels);
C_train = confusionmat(T_train, Y_train_labels);

% 创建数字到罗马数字的映射关系
label_mapping = containers.Map({1, 2, 3, 4, 5}, {'I', 'II', 'III', 'IV', 'V'});

% 将数字标签转换为罗马数字标签
T_train_roman = cellfun(@(x) label_mapping(x), num2cell(T_train), 'UniformOutput', false);
Y_train_roman = cellfun(@(x) label_mapping(x), num2cell(Y_train_labels), 'UniformOutput', false);
T_test_roman = cellfun(@(x) label_mapping(x), num2cell(T_test), 'UniformOutput', false);
Y_test_roman = cellfun(@(x) label_mapping(x), num2cell(Y_test_labels), 'UniformOutput', false);

% 绘制训练集的混淆矩阵可视化
figure;
cm_train = confusionchart(T_train_roman, Y_train_roman);
cm_train.ColumnSummary = 'column-normalized';
cm_train.RowSummary = 'row-normalized';


% 绘制测试集的混淆矩阵可视化
figure;
cm_test = confusionchart(T_test_roman, Y_test_roman);
cm_test.ColumnSummary = 'column-normalized';
cm_test.RowSummary = 'row-normalized';

figure
plot(1:M, T_train, 'r-*', 1:M, Y_train_labels, 'b-o', 'LineWidth', 1)
legend('True value', 'Predicted value', 'FontSize', 12); 
xlabel('Prediction Sample','FontSize', 12)
ylabel('Prediction Result','FontSize', 12)
title('Training and Validation Accuracy:92.42%','FontSize', 12)
xlim([1, M])
grid on

figure;
plot(1:N, T_test, 'r-*', 1:N, Y_test_labels, 'b-o', 'LineWidth', 1)

% 设置标签和标题的字体大小
xlabel('Prediction Sample', 'FontSize', 12); % 设置横轴标签字体大小为 14
ylabel('Prediction Result', 'FontSize', 12); % 设置纵轴标签字体大小为 14
title('Testing Accuracy: 91.11%', 'FontSize', 12); % 设置标题字体大小为 16

% 设置图例
legend('True value', 'Predicted value', 'FontSize', 12); % 设置图例字体大小为 12

xlim([1, N])
grid on