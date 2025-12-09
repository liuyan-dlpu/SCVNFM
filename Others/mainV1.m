% %%%%%%%%%%%%% 交叉验证 %%%%%%%%%%%%%%
% clear all
% clc
% close all
%   
% %% 读入数据
% % % Landsat Satellite
% % datatrain = readtable('./Datasets/sat_train.txt'); % 自动检测分隔符
% % datatest = readtable('./Datasets/sat_test.txt'); % 自动检测分隔符
% % data = [datatrain;datatest];
% % data = data{:,:};
% 
% 
% % % waveform
% % filePath = './Datasets/waveform.data'; % 指定文件路径
% % opts = detectImportOptions(filePath, 'FileType', 'text'); % 指定文件类型为文本 
% % dataTable = readtable(filePath, opts);
% % data = dataTable{:, :}; % 特征
% % 
% % % water
% % filePath = './Datasets/water-treatment.data'; % 指定文件路径
% % opts = detectImportOptions(filePath, 'FileType', 'text'); % 指定文件类型为文本 
% % dataTable = readtable(filePath, opts);
% % data = dataTable{:, :}; % 特征
% 
% 
% 
% % % Breast Cancer Wisconsin
% % load('./Datasets/wisconsin_no_mis_vals.mat') ;
% % data = wisconsin_no_mis_vals;
% 
% % % Heart
% % load('heart.mat') 
% % data = heart;
% 
% % % Liver Disorders
% % load('liver.mat');
% % data = liver;
% 
% % % Sonar
% % load('./Datasets/Sonar.mat');
% % data = Sonar;
% 
% % Wine
% load('./Datasets/wine.mat');
% data = wine;
% 
% 
% 
% 
% 
% 
% %% n次k折交叉验证
% n = 10;
% k = 5;
% 
% [SampleNum, N] = size(data);       % the number of all samples
% result = zeros(n, 3);
% for j = 1:n
% data = data(randperm(SampleNum),:); % 打乱
% 
% X = data(:, 1:N-1);  % 假设最后一列为标签
% X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化
% 
% y = data(:, N);  % 标签
% % y(y == 1) = 0;
% % y(y == 2) = 1;
% 
% y(y == 1) = 0;
% y(y == 2) = 1;
% y(y == 3) = 2;
% % y(y == 4) = 4;
% % y(y == 5) = 5;
% % y(y == 7) = 6;
% 
% cv = cvpartition(SampleNum, 'KFold', k);
% 
% % 初始化误差数组
% results = zeros(k, 3);
% 
%     for i = 1:k
%         % 训练集和测试集的索引
%         trainIdx = cv.training(i);
%         testIdx = cv.test(i);
% 
%         % 训练数据
%         X_train = X(trainIdx, :);
%         y_train = y(trainIdx);
% 
%         % 测试数据
%         X_test = X(testIdx, :);
%         y_test = y(testIdx);
% 
%         % 输入SCFNFM，读取结果
%         results(i, :) = SCVNFMV1(X_train, y_train, X_test, y_test);
% %         results(i, :) = SCVNFMV1_NoL2(X_train, y_train, X_test, y_test);
% %         results(i, :) = SCVNFMV1_NoMomentum(X_train, y_train, X_test, y_test);
% %         results(i, :) = SCVNFMV1_NoMomentumL2(X_train, y_train, X_test, y_test);
% 
% 
% 
%     end
% result(j, :) = sum(results,1) / k;
% end
% fprintf('%d次%d折交叉验证平均测试准确率为%.2f%%\n',  n, k, sum(result(:,3)) / n);
% 
% 
% 







%%%%%%%%%%%%% n次实验的平均值 %%%%%%%%%%%%%%
clear all
clc
close all

%% 读入数据

% % waveform
% filePath = './datasets/waveform.data'; % 指定文件路径
% opts = detectImportOptions(filePath, 'FileType', 'text'); % 指定文件类型为文本 
% dataTable = readtable(filePath, opts);
% 
% SampleNum = height(dataTable); % 获取数据个数
% dataTable = dataTable(randperm(SampleNum), :); % 打乱数据表 
% 
% X = dataTable{:, 1:end-1}; % 特征
% X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化
% y = dataTable{:, end};     % 标签三分类，0，1，2
% 
% [~, N] = size(X);           % 获取特征个数N


% % Landsat Satellite
% datatrain = readtable('./Datasets/sat_train.txt'); % 自动检测分隔符
% datatest = readtable('./Datasets/sat_test.txt'); % 自动检测分隔符
% TrainNum = height(datatrain); % 获取数据个数
% dataTable = [datatrain;datatest];
% SampleNum = height(dataTable); % 获取数据个数
% X = dataTable{:, 1:end-1}; % 特征
% X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化
% y = dataTable{:, end};     % 标签三分类，0，1，2
% y(y == 1) = 0;
% y(y == 2) = 0.2;
% y(y == 3) = 0.4;
% y(y == 4) = 0.6;
% y(y == 5) = 0.8;
% y(y == 7) = 1;
% [~, N] = size(X);           % 获取特征个数N
% x = X(1:TrainNum,:);
% O = y(1:TrainNum,:);
% Test1 = X(TrainNum+1:end,:);
% OTest = y(TrainNum+1:end,:);

% Breast Cancer Wisconsin
load('./Datasets/wisconsin_no_mis_vals.mat') ;
data = wisconsin_no_mis_vals;

% % Heart
% load('heart.mat') 
% data = heart;

% % Liver Disorders
% load('liver.mat');
% data = liver;

% % Sonar
% load('./Datasets/Sonar.mat');
% data = Sonar;

% % Wine
% load('wine.mat');
% data = wine;

%% 数据预处理
[SampleNum, N] = size(data);       % the number of all samples
data = data(randperm(SampleNum),:); % 打乱

X = data(:, 1:N-1);  % 假设最后一列为标签
X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化

y = data(:, N);  % 标签
% y(y == 1) = 0;
% y(y == 2) = 1;

y(y == 2) = 0;
y(y == 4) = 1;
% y(y == 3) = 2;
% y(y == 4) = 4;
% y(y == 5) = 5;
% y(y == 7) = 6;

trainSize = round(0.7 * SampleNum);

X_train = X(1:trainSize,:);
y_train = y(1:trainSize,:);

X_test = X(trainSize+1:end,:);
y_test = y(trainSize+1:end,:);

n = 20;
results = zeros(n, 4);
for j = 1:n
    % 输入SCFNFM，读取结果
    % results(j, :) = SCVNFMV1(X_train, y_train, X_test, y_test);
%     results(j, :) = SCVNFMV1_NoL2(X_train, y_train, X_test, y_test);
    % results(j, :) = SCVNFMV1_NoMomentum(X_train, y_train, X_test, y_test);
%     results(j, :) = SCVNFMV1_NoMomentumL2(X_train, y_train, X_test, y_test);

    results(j, :) = SCVNFMV1_t(X_train, y_train, X_test, y_test);
end
fprintf('%d次实验平均测试准确率为%.2f%%\n',  n, sum(results(:,3)) / n);
save('BreastCancer_SCVNFM_98.02.mat')
sum(results)/n




