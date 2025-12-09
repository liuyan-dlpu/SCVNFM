% 交叉验证 %
function results = SCVNFMV1_NoL2(x, O, Test1, OTest) % X 是特征矩阵，y 是标签向量
    [SampleNum, Indim] = size(x);       % the number of all samples
    TestNum = length(OTest);            % the number of all samples

    %%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
    Num_subset = 5;                     % 各语言变量对应的模糊子集的个数
    HiddenUnit = Num_subset;            % 隐节点数
    err = 0;                            % 初始误差
    E0 = 0.01;                          % 误差标准
    MaxEpochs = 1000;                  % 最大迭代步数
    epochs = 1;                         % 步数
    Ir = 0.001;                        % 学习率
    Tau = 0.0009;                       % 动量常系数
    e = [0,0];                          % 存储误差
    NormW = [0,0];                      % 存储梯度范数
%     a = rand(Indim,Num_subset);         % 高斯函数的初始中心值
%     b = rand(Indim,Num_subset);         % 高斯函数的初始宽度值     
%     w = rand(Num_subset,Indim+1);       % 结论参数的初始值
    a=0.1*rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
    b=0.5*rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
    w=0.1*rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值

    l = 0;                              % 跳出开关
    Delte_a = a.*0;                     % 中心梯度初值
    Delte_b = b.*0;                     % 宽度梯度初值
    Delte_w = w.*0;                     % 结论参数梯度初值
    t = 0;
    tic;

    NetworkOut = zeros(SampleNum,1);
    max_rightratioTest = 0;             % 最佳测试准确率
    best_rightratiotrain = 0;           % 对应的最佳训练准确率
	bestepoch = 0;


    %%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
    while(epochs < MaxEpochs && l == 0)
        %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
        Gradient_a = a.*0;              % 中心梯度初值
        Gradient_b = b.*0;              % 宽度梯度初值
        Gradient_w = w.*0;              % 结论参数梯度初值
        Gradient1_a = a.*0;             % 加L2中心梯度初值
        Gradient1_b = b.*0;             % 加L2宽度梯度初值
        Gradient1_w = w.*0;             % 加L2结论参数梯度初值
        Hidden = ones(SampleNum,Num_subset); % 隐节点初值
        Hiddensum = zeros(SampleNum,Num_subset);
        err = 0;
        NetworkOut = NetworkOut * 0;

        %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%修改后算法
        for i = 1:SampleNum
            for j = 1:Num_subset
                for k = 1:Indim
                    Hidden(i,j) = real(exp(-(x(i,k) - a(k,j)) * conj(x(i,k) - a(k,j)) * (b(k,j) * b(k,j)))) * Hidden(i,j);
                    Hiddensum(i,j) = w(j,k) * x(i,k) + Hiddensum(i,j);
                end
                Hiddensum(i,j) = Hiddensum(i,j) + w(j,Indim+1);
            end
            for j = 1:Num_subset
                NetworkOut(i) = Hidden(i,j) * Hiddensum(i,j) + NetworkOut(i);
            end
        end

        %%%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
        for i = 1:SampleNum
            err = 1/SampleNum * (O(i) - NetworkOut(i)) * conj(O(i) - NetworkOut(i)) + err;
        end
        err = real(err);
        e(epochs) = err;

        %%%% 计算训练准确度 %%%%
        NetworkOut111 = NetworkOut;
        NetworkOut(find(NetworkOut < 0.5)) = 0;
        NetworkOut(find(NetworkOut >= 0.5)) = 1;
%         NetworkOut(find(NetworkOut < 1.5)) = 1;
%         NetworkOut(find(NetworkOut >= 1.5 & NetworkOut < 2.5)) = 2;
%         NetworkOut(find(NetworkOut >= 2.5 & NetworkOut < 3.5)) = 3;
%         NetworkOut(find(NetworkOut >= 3.5 & NetworkOut < 4.5)) = 4;
%         NetworkOut(find(NetworkOut >= 4.5 & NetworkOut < 5.5)) = 5;
%         NetworkOut(find(NetworkOut >= 5.5)) = 6;

        rightnumber = sum(O == NetworkOut);
        rightratiotrain = rightnumber / SampleNum * 100;

        %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
        if(err < E0)
            l = 1;
        end

        %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
        for j = 1:Num_subset
            for k = 1:Indim  
                for i = 1:SampleNum
                    Delpublic = O(i) - NetworkOut(i);
                    Gradient_w(j,k) = Delpublic * Hidden(i,j) * x(i,k) + Gradient_w(j,k);
                end
            end 
        end

        for j = 1:Num_subset
            for i = 1:SampleNum
                Delpublic = O(i) - NetworkOut(i);
                Gradient_w(j,Indim+1) = Delpublic * Hidden(i,j) + Gradient_w(j,Indim+1);
            end 
        end 

        %%%%%%%%%%%% 动量 %%%%%%%%%%%%%%%%%%%%
        Alpha_a = 0;
        Alpha_b = 0;
        Alpha_w = 0;
        if norm(Delte_a) > 0.0001
            Alpha_a = Tau * norm(Gradient_a) / norm(Delte_a);
        end
        if norm(Delte_b) > 0.0001
            Alpha_b = Tau * norm(Gradient_b) / norm(Delte_b);
        end
        if norm(Delte_w) > 0.0001
            Alpha_w = Tau * norm(Gradient_w) / norm(Delte_w);
        end
        Delte_a = Ir * Gradient_a + Alpha_a * Delte_a;
        Delte_b = -Ir * Gradient_b + Alpha_b * Delte_b;
        Delte_w = Ir * Gradient_w + Alpha_w * Delte_w;
        a = a + Delte_a;
        b = b + Delte_b;
        w = w + Delte_w;

        %%%%%%%%%%%%%%%%%计算梯度的范数%%%%%%%%%%%%%
        DeltaA = reshape(Gradient_a, 1, Indim * Num_subset);
        DeltaB = reshape(Gradient_b, 1, Indim * Num_subset);
        DeltaW = reshape(Gradient_w, 1, Num_subset * (Indim + 1));
        Norm_W = norm([DeltaA DeltaB DeltaW]);
        NormW(epochs) = Norm_W;

        %%%%%%%%%%%% 测试部分 %%%%%%%%%%%%%%%%%%%%%%%
        Hidden = ones(TestNum, Num_subset);
        Hiddensum = zeros(TestNum, Num_subset);
        TestOut = zeros(TestNum, 1); 

        for i = 1:TestNum
            for j = 1:Num_subset
                for k = 1:Indim
                    Hidden(i,j) = real(exp(-(Test1(i,k) - a(k,j)) * conj(Test1(i,k) - a(k,j)) * (b(k,j) * b(k,j)))) * Hidden(i,j);
                    Hiddensum(i,j) = w(j,k) * Test1(i,k) + Hiddensum(i,j);
                end
                Hiddensum(i,j) = w(j, Indim+1) + Hiddensum(i,j);
            end
            for j = 1:Num_subset
                TestOut(i) = Hidden(i,j) * Hiddensum(i,j) + TestOut(i);
            end
        end
        TestOut111 = TestOut;
        %%%%%%%%%%%%%%%%%% 计算测试误差与准确率 %%%%%%%%%%%%%%
%         TestOut(find(TestOut < 1.5)) = 1;
%         TestOut(find(TestOut >= 1.5 & TestOut < 2.5)) = 2;
%         TestOut(find(TestOut >= 2.5 & TestOut < 3.5)) = 3;
%         TestOut(find(TestOut >= 3.5 & TestOut < 4.5)) = 4;
%         TestOut(find(TestOut >= 4.5 & TestOut < 5.5)) = 5;
%         TestOut(find(TestOut >= 5.5)) = 6;
        TestOut(find(TestOut < 0.5)) = 0;
        TestOut(find(TestOut >= 0.5)) = 1;

        rightnumberTest = sum(OTest == TestOut);
        rightratioTest = rightnumberTest / TestNum * 100;

        % 如果当前测试准确率更高，则保存最佳测试准确率和对应的训练准确率
        if rightratioTest >= max_rightratioTest
            bestepoch = epochs; 
            max_rightratioTest = rightratioTest;
            best_rightratiotrain = rightratiotrain;
        end

        epochs = epochs + 1;
    end

    t = toc;
%     plot(1:MaxEpochs-1,e)

%     sprintf('最佳测试准确率=%0.2f%% 对应训练准确率=%0.2f%%', max_rightratioTest, best_rightratiotrain
    results = [bestepoch, best_rightratiotrain, max_rightratioTest];
end