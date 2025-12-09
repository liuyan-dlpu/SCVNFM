function [Data] = LoadData_Wind_finally()
%%%% 每六个取平均
[num,txt,raw] = xlsread('Wind_finally.xlsx');
data = [];
clumn = size(num, 1);
if mod(clumn, 6) ~= 0
    error('错误: 数据行不是6的倍数');
end
m = 0;
while m < clumn
    part_mat = num(m+1:m+6,1:10);
    temp = sum(part_mat, 1) / 6;
    data = [data; temp];
    m = m + 6;
end
% % % % disp(data);
data;
%% 获取原始Planes 数据集， 其中最后一列表示 输出 本算法未考虑car name， 忽略掉6条未知数据
%% 
OriWind_finally = data;

 %%%% 先把第一列变成最后一列，然后转置，变成 8 行 392列，最后一行表示理想输出 
OriWind_finally = [OriWind_finally(:,1:end-2) OriWind_finally(:,end-1:end-1)]';   % 先把第一列变成最后一列，然后转置，变成 8 行 392列，最后一行表示理想输出 

[OriWind_finally,PS] = mapminmax(OriWind_finally);   %%%%%%% 对原始数据归一化，转置以后变成 8 行 392列，最后一行表示输出,行 表示属性，列 表示样本个数。

%%%风速预测
SamInAll = OriWind_finally(1:end-1,:);   %样本输入，OriautoMPG 前end-1行，7 行 392 列
SamOutAll = OriWind_finally(end:end,:);  %样本输出，OriautoMPG 最后一行
%%%风速预测

%%随机选择输入输出样本个数
RateTr = 0.75;    % 设置训练样本率, 随机选取
NumSamAll = size(SamInAll,2); % 整体数据集包含的样本数，共 392 个数
RmPerm = randperm(NumSamAll);      % 产生样本数的随机排列，产生一个向量，包含 392 个元素，392个元素是从 1-392 之间随机产生的。

NumTr = round(NumSamAll*RateTr);    % 训练样本数，四舍五入函数
NumTe = NumSamAll-NumTr;            % 测试样本数 
%%随机选择输入输出样本个数

%%% 这个地方表示什么意思
TrSamIn = SamInAll(:,RmPerm(1:NumTr));         % 从 392 列中，随机选择 NumTr 列，作为训练输入
TrSamOut = SamOutAll(:,RmPerm(1:NumTr));       % 训练样本集的输出 

TeSamIn = SamInAll(:,RmPerm(NumTr+1:NumSamAll));    % 测试样本集(输入) 392列中选完 NumTr 列后，剩余的为测试样本集
TeSamOut = SamOutAll(:,RmPerm(NumTr+1:NumSamAll));     % 测试样本集(输出)


%%　用于函数调用
Data = struct();
Data.TrSamIn = TrSamIn';
Data.TrSamOut = TrSamOut';

Data.TeSamIn = TeSamIn';
Data.TeSamOut = TeSamOut';
Data.PS = PS;    
%%　用于函数调用

