function [Data] = LoadData_HTRU2()
clear
clc

% 读取文件
f = fopen('hTRU2.data');
hTRU2 = textscan(f,'%f%f%f%f%f%f%f%f%s','delimiter',',');
fclose(f);
[~,n] = size(hTRU2);

% 读取特征，并处理
for i = 1:n-1
    datax(:,i) = hTRU2{1,i};
end
[SampleNum,FeaNum] = size(datax);

datax = mapminmax(datax',-1,1);       
datax = datax';

% 读取标签
datay = zeros(SampleNum,1);
for i = 1:SampleNum
    if strcmp(hTRU2{1,n}{i,1},'0')
        datay(i) = 0;
    else
        datay(i) = 1;
    end
end

% 分训练集和测试集，大概随机选取2/3数据用于训练集
pos = randperm(SampleNum);            % Shuffle the order of samples
datax = datax(pos,:);    
datay = datay(pos,:);
TrainNum = 12528;

TrSamIn=datax(1:TrainNum,:);    %%% 训练集输入
TrSamOut=datay(1:TrainNum,:);   %%%%训练集输出
TeSamIn=datax(TrainNum +1:SampleNum,:); %%%测试集输入
TeSamOut=datay(TrainNum +1:SampleNum,:);%%%测试集输出

%%　用于函数调用
Data = struct();
Data.TrSamIn = TrSamIn;
Data.TrSamOut = TrSamOut;

Data.TeSamIn = TeSamIn;
Data.TeSamOut = TeSamOut;
%%　用于函数调用
end
