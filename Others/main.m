%%%%%%%%%%%%% 交叉验证 %%%%%%%%%%%%%%
clear all
clc
close all
  
%% 读入数据
% % Breast Cancer Wisconsin
% load('wisconsin_no_mis_vals.mat') ;
% data = wisconsin_no_mis_vals;

% % Heart
% load('heart.mat') 
% data = heart;

% % Liver Disorders
% load('liver.mat');
% data = liver;

% % Sonar
% load('Sonar.mat');
% data = Sonar;

% % Wine
% load('wine.mat');
% data = wine;




[SampleNum, N] = size(data);       % the number of all samples

%% n次k折交叉验证
n = 10;
k = 5;

result = zeros(n, 3);
for j = 1:n
data = data(randperm(SampleNum),:); % 打乱

X = data(:, 1:N-1);  % 假设最后一列为标签
X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化

y = data(:, N);  % 标签
y(y == 1) = 0;
y(y == 2) = 1;
y(y == 2) = 2;

cv = cvpartition(SampleNum, 'KFold', k);

% 初始化误差数组
results = zeros(k, 3);

    for i = 1:k
        % 训练集和测试集的索引
        trainIdx = cv.training(i);
        testIdx = cv.test(i);

        % 训练数据
        X_train = X(trainIdx, :);
        y_train = y(trainIdx);

        % 测试数据
        X_test = X(testIdx, :);
        y_test = y(testIdx);

        % 输入SCFNFM，读取结果
        results(i, :) = SCVNFMV1(X_train, y_train, X_test, y_test);


    end
result(j, :) = sum(results,1) / k;
end
fprintf('%d次%d折交叉验证平均测试准确率为%.2f%%\n',  n, k, sum(result(:,3)) / n);


% %%%%%%%%%%%%% n次实验的平均值 %%%%%%%%%%%%%%
% clear all
% clc
% close all
%   
% n = 5;
% results = zeros(n, 2);
% for j = 1:n
%     % 输入SCFNFM，读取结果
%     results(j, :) = example3_wine();
% end
% fprintf('%d次实验平均训练准确率为%.2f%%，测试准确率为%.2f%%\n',  n, sum(results(:,1)) / n, sum(results(:,2)) / n);


