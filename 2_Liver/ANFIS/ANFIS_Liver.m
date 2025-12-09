%%%%%%%%%% ANFIS Implementation Without Toolbox (No Regularization/Momentum)
clear;
clc;
ExampleNum = 10; % 实验次数
Allresult = zeros(ExampleNum,4);

% 选择要详细显示的实验编号
for gogo = 1:ExampleNum
    %%%%%%%%%%%%%%%%%%%%%%%%% Datas Processing %%%%%%%%%%%%%%%%%%%%%%%
    data = load("Liver.txt");
    
    %%%%%%%%%%% 数据处理 %%%%%%%%%%%%%%%%%%%%%%%%
    InputNeuronsNum = size(data,2)-1;           %   (属性数)列数 - 1 （原因是最后一列为分类列）
    DataNum = size(data,1);                     %   (样本数)行数
    sorted_target = sort(data(:,InputNeuronsNum+1));  %   有序标签
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:DataNum                           %   确定数据集的类别数量
        if sorted_target(i,1) ~= label(1,j)     %   如果 排列标签(1,2)不=标签(1,1)    //matlab中 ~= 表 不等于
            j=j+1;                              %   j+1 = 2
            label(1,j) = sorted_target(i,1);    %   标签(1,2)=排列标签(1,2)
        end
    end
    ClassNum=j;                                 %   类别数量
    OutputNeuronsNum=ClassNum;                  %   输出神经元数量
    rowrank = randperm(DataNum);                %   打乱矩阵
    data = data(rowrank,:);
    AttributeData = data(:,1:InputNeuronsNum);  %   属性
    TableData = data(:,InputNeuronsNum+1);      %   标签
    TrainNum = round(0.7*DataNum);                     %   训练样本选60%
    TestNum = DataNum - TrainNum;               %   测试样本为剩余
    Train_Atttibute = AttributeData(1:TrainNum,:);              %   训练样本属性
    % Train_Atttibute = Train_Atttibute';
    Train_Table = TableData(1:TrainNum,:);                      %   训练样本标签
    Test_Atttibute = AttributeData((TrainNum+1):DataNum,:);     %   测试样本属性
    % Test_Atttibute = Test_Atttibute';
    Test_Table = TableData((TrainNum+1):DataNum,:);             %   测试样本标签
    
    %%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
    Indim=InputNeuronsNum;                      %%%%输入样本维数
    Outdim=1;                     %%%%输出样本维数
    
    x=Train_Atttibute;    %%% 训练集输入
    O=Train_Table;   %%%%训练集输出
    SampleNum = TrainNum;
    NetworkOut=zeros(1,SampleNum);
    
    Test1=Test_Atttibute; %%%测试集输入
    % TestNum=size(Test1,1);
    OTest=Test_Table;%%%测试集输出
    
    %%%%%%%%%%%  参数设置 %%%%%%%%%%%%%%%%%%%%%%%%
    Num_subset = 4;             % 各输入变量对应的模糊子集的个数
    HiddenUnit = Num_subset;    % 隐节点数（规则数）
    err = 0;                    % 初始误差
    E0 = 0.01;                  % 误差标准
    MaxEpochs = 2000;           % 最大迭代步数
    epochs = 1;                 % 步数
    Ir = 0.5;                 % 学习率
    
    % 记录每个epoch的结果
    Traine = zeros(1, MaxEpochs);        % 训练误差
    Teste = zeros(1, MaxEpochs);         % 测试误差
    NormW = zeros(1, MaxEpochs);         % 梯度范数
    TrainOutputs = cell(1, MaxEpochs);   % 训练输出
    TestOutputs = cell(1, MaxEpochs);    % 测试输出
    
    % 初始化参数
    a = 0.1 * rand(Indim, Num_subset);     % 高斯函数的初始中心值
    b = 0.5 * rand(Indim, Num_subset);     % 高斯函数的初始宽度值     
    w = 0.1 * rand(Num_subset, Indim + 1); % 结论参数的初始值
    
    l = 0;                                  % 跳出开关
    
    % 记录最佳结果
    Best_TestAcc = 0;             % 最佳测试准确率
    Best_TrainAcc = 0;           % 对应的最佳训练准确率
    Best_Epoch = 0;
    
    tic;
    
    %%%%%%%%%%%%%%%%%%%%%%  训练循环  %%%%%%%%%%%%%%%%%%%%
    while(epochs <= MaxEpochs && l == 0)
        
        %%%%%%%%%%%%%%%%%%%%% 重置梯度 %%%%%%%%%%%%% 
        Gradient_a = zeros(size(a));
        Gradient_b = zeros(size(b));
        Gradient_w = zeros(size(w));
        
        % 前向传播：计算网络输出
        NetworkOut = zeros(SampleNum, 1);
        FireStrength = ones(SampleNum, Num_subset);  % 规则触发强度
        RuleOutput = zeros(SampleNum, Num_subset);   % 规则输出
        
        for i = 1:SampleNum
            % 计算每条规则的触发强度（使用高斯隶属函数）
            for j = 1:Num_subset
                fire_strength = 1;
                for k = 1:Indim
                    % 高斯隶属函数
                    mu = exp(-(x(i,k) - a(k,j))^2  * b(k,j)^2);
                    fire_strength = fire_strength * mu;
                end
                FireStrength(i,j) = fire_strength;
                
                % 计算规则输出（一阶TSK模型）
                rule_out = w(j, Indim+1);  % 常数项
                for k = 1:Indim
                    rule_out = rule_out + w(j,k) * x(i,k);
                end
                RuleOutput(i,j) = rule_out;
            end
            
            % 归一化触发强度
            total_fire = sum(FireStrength(i,:));
            if total_fire > 0
                NormalizedFire = FireStrength(i,:) / total_fire;
            else
                NormalizedFire = FireStrength(i,:);
            end
            
            % 计算最终输出（加权平均）
            NetworkOut(i) = sum(NormalizedFire .* RuleOutput(i,:));
        end
        
        %%%%%%%%%%%%%%%%%%%% 计算训练误差 %%%%%%%%%%%%%%
        err = 0;
        for i = 1:SampleNum
            err = err + (O(i) - NetworkOut(i))^2;
        end
        err = err / SampleNum;
        Traine(epochs) = err;

         %%% 训练准确率
         for TF = 1 : SampleNum
             if NetworkOut(TF)<1.5
                 NetworkOut(TF) = 1;
             elseif NetworkOut(TF)>=1.5
                 NetworkOut(TF) = 2;
             end
         end
    
         Trainrightnumber=0;
         for i=1:SampleNum
             if O(i)==NetworkOut(i)
                 Trainrightnumber=Trainrightnumber+1;
             end
         end
         TrainAcc=Trainrightnumber/SampleNum*100;
         TrainAccSave(epochs) = TrainAcc;
        
        %%%%%%%%%%%%%%%%%%% 计算测试误差 %%%%%%%%%%%%%%
        TestOut = zeros(TestNum, 1);
        for i = 1:TestNum
            TestFireStrength = ones(1, Num_subset);
            TestRuleOutput = zeros(1, Num_subset);
            
            for j = 1:Num_subset
                fire_strength = 1;
                for k = 1:Indim
                    mu = exp(-(Test1(i,k) - a(k,j))^2 * b(k,j)^2);
                    fire_strength = fire_strength * mu;
                end
                TestFireStrength(j) = fire_strength;
                
                rule_out = w(j, Indim+1);
                for k = 1:Indim
                    rule_out = rule_out + w(j,k) * Test1(i,k);
                end
                TestRuleOutput(j) = rule_out;
            end
            
            total_fire = sum(TestFireStrength);
            if total_fire > 0
                NormalizedFire = TestFireStrength / total_fire;
            else
                NormalizedFire = TestFireStrength;
            end
            
            TestOut(i) = sum(NormalizedFire .* TestRuleOutput);
        end
        
        TestErr = 0;
        for i = 1:TestNum
            TestErr = TestErr + (OTest(i) - TestOut(i))^2;
        end
        TestErr = TestErr / TestNum;
        Teste(epochs) = TestErr;

        for TTF = 1 : TestNum
             if TestOut(TTF)<1.5
                 TestOut(TTF) = 1;
             elseif TestOut(TTF)>=1.5
                 TestOut(TTF) = 2;
             end
         end
       
         Trainrightnumber=0;
         for i=1:TestNum
             if OTest(i)==TestOut(i)
                 Trainrightnumber=Trainrightnumber+1;
             end
         end
         TestAcc=Trainrightnumber/TestNum*100;
         TestAccSave(epochs) = TestAcc;

        
        % 记录当前epoch的输出结果
        TrainOutputs{epochs} = NetworkOut;
        TestOutputs{epochs} = TestOut;
        
        %%%%%%%%%%%%%%%%%%% 反向传播：计算梯度 %%%%%%%%%%%%%%%%%
        for j = 1:Num_subset
            for k = 1:Indim
                for i = 1:SampleNum
                    % 计算输出误差
                    error_term = O(i) - NetworkOut(i);
                    
                    % 计算归一化触发强度对最终输出的导数
                    total_fire = sum(FireStrength(i,:));
                    normalized_fire_j = FireStrength(i,j) / total_fire;
                    
                    % 计算结论参数梯度
                    Gradient_w(j,k) = Gradient_w(j,k) + error_term * normalized_fire_j * x(i,k);
                    Gradient_w(j,Indim+1) = Gradient_w(j,Indim+1) + error_term * normalized_fire_j;
                    
                    % 计算前提参数梯度（简化版本）
                    rule_diff = RuleOutput(i,j) - NetworkOut(i);
                    Gradient_a(k,j) = Gradient_a(k,j) + error_term * rule_diff * ...
                        normalized_fire_j * (1 - normalized_fire_j) * ...
                        (x(i,k) - a(k,j)) * (b(k,j)^2) * 2;
                    
                    Gradient_b(k,j) = Gradient_b(k,j) + error_term * rule_diff * ...
                        normalized_fire_j * (1 - normalized_fire_j) * ...
                        (x(i,k) - a(k,j))^2 * b(k,j) * (-2);
                end
                
                % 归一化梯度
                Gradient_w(j,k) = Gradient_w(j,k) / SampleNum;
                Gradient_a(k,j) = Gradient_a(k,j) / SampleNum;
                Gradient_b(k,j) = Gradient_b(k,j) / SampleNum;
            end
            Gradient_w(j,Indim+1) = Gradient_w(j,Indim+1) / SampleNum;
        end
        
        % 计算梯度范数
        GradientNorm = norm([Gradient_a(:); Gradient_b(:); Gradient_w(:)]);
        NormW(epochs) = GradientNorm;
        
        % 直接使用梯度下降更新参数（无动量）
        a = a + Ir * Gradient_a;
        b = b + Ir * Gradient_b;
        w = w + Ir * Gradient_w;
        
        % 确保宽度参数为正
        %b = max(b, 0.01);
        
        % 记录最佳结果
        if TestAcc >= Best_TestAcc
         Best_Epoch = epochs;
         Best_TestAcc = TestAcc;
         Best_TrainAcc = TrainAcc; 
        end
        
        % 每100轮显示进度
        if mod(epochs, 100) == 0
            fprintf('Epoch %d: TrainAcc=%.6f, TestAcc=%.6f, GradNorm=%.6f\n', ...
                epochs, Best_TrainAcc,Best_TestAcc, GradientNorm);
        end
        
        epochs = epochs + 1;
    end
    
    t = toc;
    
    % 存储结果
    Allresult(gogo,:) = [Best_TrainAcc, Best_TestAcc, Best_Epoch, t];
    
    fprintf('Experiment %d: Best_TrainAcc=%.6f, Best_TestAcc=%.6f, Epochs=%d, Time=%.2fs\n', ...
            gogo, Best_TrainAcc, Best_TestAcc, Best_Epoch, t);
    
    % 如果是详细显示的实验，绘制详细结果
end

%%%%%%%%%%% 计算平均结果 %%%%%%%%%%%%%%%%%%%%%%%%
Meanruselt = mean(Allresult, 1);
Mean_TrainAcc = Meanruselt(1);
Mean_TestAcc = Meanruselt(2);
Mean_Epoch = Meanruselt(3);
Mean_Time = Meanruselt(4);

fprintf('\n=== Final ANFIS Results ===\n');
fprintf('Average Training MSE: %.6f\n', Mean_TrainAcc);
fprintf('Average Testing MSE: %.6f\n', Mean_TestAcc);
fprintf('Average Epochs: %.0f\n', Mean_Epoch);
fprintf('Average Training Time: %.2f seconds\n', Mean_Time);
