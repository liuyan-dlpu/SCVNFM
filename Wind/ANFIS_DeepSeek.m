%%%%%%%%%% ANFIS Implementation Without Toolbox (No Regularization/Momentum)
clear;
clc;
ExampleNum = 10; % 实验次数
AllErr = zeros(ExampleNum,4);

% 选择要详细显示的实验编号
DetailedExp = 0; % 显示第1次实验的详细结果

for gogo = 1:ExampleNum
    %%%%%%%%%%%%%%%%%%%%%%%%% Datas Processing %%%%%%%%%%%%%%%%%%%%%%%
    load('Data.mat')
    
    %%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
    x = Data.TrSamIn;       % 训练集输入
    O = Data.TrSamOut;      % 训练集输出
    Test1 = Data.TeSamIn;   % 测试集输入
    OTest = Data.TeSamOut;  % 测试集输出
    
    % 确保数据为实数
    if ~isreal(x)
        x = real(x);
    end
    if ~isreal(O)
        O = real(O);
    end
    if ~isreal(Test1)
        Test1 = real(Test1);
    end
    if ~isreal(OTest)
        OTest = real(OTest);
    end
    
    Indim = size(x,2);      % 输入样本维数
    Outdim = 1;             % 输出样本维数
    SampleNum = size(x,1);
    TestNum = size(Test1,1);
    
    %%%%%%%%%%%  参数设置 %%%%%%%%%%%%%%%%%%%%%%%%
    Num_subset = 10;             % 各输入变量对应的模糊子集的个数
    HiddenUnit = Num_subset;    % 隐节点数（规则数）
    err = 0;                    % 初始误差
    E0 = 0.01;                  % 误差标准
    MaxEpochs = 10000;           % 最大迭代步数
    epochs = 1;                 % 步数
    Ir = 0.003;                 % 学习率
    
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
    Best_TestErr = 100;
    Best_TrainErr = 100;
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
                    mu = exp(-(x(i,k) - a(k,j))^2 / (2 * b(k,j)^2));
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
        
        %%%%%%%%%%%%%%%%%%% 计算测试误差 %%%%%%%%%%%%%%
        TestOut = zeros(TestNum, 1);
        for i = 1:TestNum
            TestFireStrength = ones(1, Num_subset);
            TestRuleOutput = zeros(1, Num_subset);
            
            for j = 1:Num_subset
                fire_strength = 1;
                for k = 1:Indim
                    mu = exp(-(Test1(i,k) - a(k,j))^2 / (2 * b(k,j)^2));
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
                        (x(i,k) - a(k,j)) / (b(k,j)^2);
                    
                    Gradient_b(k,j) = Gradient_b(k,j) + error_term * rule_diff * ...
                        normalized_fire_j * (1 - normalized_fire_j) * ...
                        (x(i,k) - a(k,j))^2 / (b(k,j)^3);
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
        b = max(b, 0.01);
        
        % 记录最佳结果
        if TestErr < Best_TestErr
            Best_Epoch = epochs;
            Best_TestErr = TestErr;
            Best_TrainErr = err;
            Best_NetworkOut = NetworkOut;
            Best_TestOut = TestOut;
        end
        
        % 每100轮显示进度
        if mod(epochs, 100) == 0
            fprintf('Epoch %d: TrainErr=%.6f, TestErr=%.6f, GradNorm=%.6f\n', ...
                epochs, err, TestErr, GradientNorm);
        end
        
        epochs = epochs + 1;
    end
    
    t = toc;
    
    % 存储结果
    AllErr(gogo,:) = [Best_TrainErr, Best_TestErr, Best_Epoch, t];
    
    fprintf('Experiment %d: Best_TrainErr=%.6f, Best_TestErr=%.6f, Epochs=%d, Time=%.2fs\n', ...
            gogo, Best_TrainErr, Best_TestErr, Best_Epoch, t);
    
    % 如果是详细显示的实验，绘制详细结果
    if gogo == DetailedExp
        % 绘制训练过程详细结果
        figure(1000);
        c = 1:epochs-1;
        
        subplot(2,3,1);
        semilogy(c, Traine(1:epochs-1), 'b-', 'LineWidth', 1.5);
        xlabel('Epoch');
        ylabel('Training Error');
        title('Training Error vs Epoch');
        grid on;
        
        subplot(2,3,2);
        semilogy(c, Teste(1:epochs-1), 'r-', 'LineWidth', 1.5);
        xlabel('Epoch');
        ylabel('Testing Error');
        title('Testing Error vs Epoch');
        grid on;
        
        subplot(2,3,3);
        semilogy(c, NormW(1:epochs-1), 'g-', 'LineWidth', 1.5);
        xlabel('Epoch');
        ylabel('Gradient Norm');
        title('Gradient Norm vs Epoch');
        grid on;
        
        subplot(2,3,4);
        plot(c, Traine(1:epochs-1), 'b-', 'LineWidth', 1.5);
        xlabel('Epoch');
        ylabel('Training Error');
        title('Training Error (Linear Scale)');
        grid on;
        
        subplot(2,3,5);
        plot(c, Teste(1:epochs-1), 'r-', 'LineWidth', 1.5);
        xlabel('Epoch');
        ylabel('Testing Error');
        title('Testing Error (Linear Scale)');
        grid on;
        
        subplot(2,3,6);
        plot(O, Best_NetworkOut, 'bo', 'MarkerSize', 4);
        hold on;
        plot([min(O), max(O)], [min(O), max(O)], 'r-', 'LineWidth', 1.5);
        xlabel('Actual Output');
        ylabel('Predicted Output');
        title(sprintf('Best Training Result (Epoch %d)', Best_Epoch));
        grid on;
        legend('Data Points', 'Ideal Fit', 'Location', 'best');
        
        sgtitle(sprintf('Detailed Results - Experiment %d', gogo));
        
        % 显示前几个epoch的详细数据
        fprintf('\n=== Detailed Results for Experiment %d (First 10 Epochs) ===\n', gogo);
        fprintf('Epoch\tTrainErr\tTestErr\tGradNorm\n');
        for i = 1:min(10, epochs-1)
            fprintf('%d\t%.6f\t%.6f\t%.6f\n', i, Traine(i), Teste(i), NormW(i));
        end
        
        % 显示最佳epoch的详细数据
        fprintf('\n=== Best Epoch (%d) Details ===\n', Best_Epoch);
        fprintf('Training Error: %.6f\n', Best_TrainErr);
        fprintf('Testing Error: %.6f\n', Best_TestErr);
        fprintf('Gradient Norm: %.6f\n', NormW(Best_Epoch));
        
        % 保存详细数据到文件
        DetailedData.Epochs = 1:epochs-1;
        DetailedData.TrainErrors = Traine(1:epochs-1);
        DetailedData.TestErrors = Teste(1:epochs-1);
        DetailedData.GradientNorms = NormW(1:epochs-1);
        DetailedData.TrainOutputs = TrainOutputs(1:epochs-1);
        DetailedData.TestOutputs = TestOutputs(1:epochs-1);
        DetailedData.BestEpoch = Best_Epoch;
        
        
    end
end

%%%%%%%%%%% 计算平均结果 %%%%%%%%%%%%%%%%%%%%%%%%
MeanMSE = mean(AllErr, 1);
Mean_TrainMSE = MeanMSE(1);
Mean_TestMSE = MeanMSE(2);
Mean_Epoch = MeanMSE(3);
Mean_Time = MeanMSE(4);

fprintf('\n=== Final ANFIS Results ===\n');
fprintf('Average Training MSE: %.6f\n', Mean_TrainMSE);
fprintf('Average Testing MSE: %.6f\n', Mean_TestMSE);
fprintf('Average Epochs: %.0f\n', Mean_Epoch);
fprintf('Average Training Time: %.2f seconds\n', Mean_Time);
