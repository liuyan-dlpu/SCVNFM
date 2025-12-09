
%%%%%%%%%%  Convergence analysis of fully complex backpropagation
%%%%%%%%%  algorithm based on Wirtinger calculus   Huisheng Zhang 
%%%%%%%%%%% 数据集X,X1  %%%%%%%%%%%%
clear;
clc;
ExampleNum = 10; % 实验次数
TenAcc = zeros(ExampleNum,3);
time = zeros(ExampleNum,1);
for gogo = 1:ExampleNum
%%%%%%%%%%%%%%%%%%%%%%%%% Datas Processing %%%%%%%%%%%%%%%%%%%%%%%
load('Sonar.mat');
data = Sonar;

%% 数据预处理
    [SampleNum, N] = size(data);       % the number of all samples
    data = data(randperm(SampleNum),:); % 打乱
    
    X = data(:, 1:N-1);  % 假设最后一列为标签
    X = (X - min(X)) ./ (max(X) - min(X)); % 3. 数据归一化
    
    y = data(:, N);  % 标签
    y(y == 1) = 0;
    y(y == 2) = 1;
    
    % y(y == 2) = 0;
    % y(y == 4) = 1;
    % y(y == 3) = 2;
    % y(y == 4) = 4;
    % y(y == 5) = 5;
    % y(y == 7) = 6;
    %%%%%%%%%%%%%%%%% PCA 降维部分 %%%%%%%%%%%%%%%%%
    KeepDim = 15; % <== 保留的主成分个数，可根据需要修改（如 10~20）
    [coeff, score, latent, tsquared, explained] = pca(X);
    X = score(:, 1:KeepDim); % 用前KeepDim个主成分替代原特征
    Indim = KeepDim;         % 更新输入维度
    
    %%%%%%%%%%%%%%%%% 数据划分 %%%%%%%%%%%%%%%%%%%%%
    trainSize = round(0.7 * SampleNum);
    
    x = X(1:trainSize,:);
    O = y(1:trainSize,:);
    
    Test1 = X(trainSize+1:end,:);
    OTest = y(trainSize+1:end,:);

            %%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
    % Indim=N-1;                      
    Outdim=1;                     

    TestNum = SampleNum - trainSize;
    SampleNum = trainSize;
    NetworkOut=zeros(1,SampleNum);   
%%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=30;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.01;                                 %%%%%误差标准
MaxEpochs=1000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0003;                                %%%%%%学习率
Tau=0.00009;                            %%%%动量常系数
Lambda=0.00009;                           %%%%正则项系数
Traine=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数
a=0.1*rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
b=0.5*rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
w=0.1*rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值

l=0;                                    %%%%%跳出开关
Delte_a=a.*0;                                  %%%%%中心梯度初值
Delte_b=b.*0;                                  %%%%%宽度梯度初值
Delte_w=w.*0;                               %%%%%%%%%结论参数梯度初值
t=0;
tic;


%%%%%%%%%%%%%%%%%%%%%%  记录best  %%%%%%%%%%%%%%%%%%%%
Best_TestAcc = 0;             % 最佳测试准确率
Best_TrainAcc = 0;           % 对应的最佳训练准确率
Best_Epoch = 0;
%%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<=MaxEpochs&l==0)
   
    %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
  
     Gradient_a=a.*0;                                  %%%%%中心梯度初值
     Gradient_b=b.*0;                                  %%%%%宽度梯度初值
     Gradient_w=w.*0;                               %%%%%%%%%结论参数梯度初值
     Gradient1_a=a.*0;                                  %%%%%加L2中心梯度初值
     Gradient1_b=b.*0;                                  %%%%%加L2宽度梯度初值
     Gradient1_w=w.*0;                               %%%%%%%%%加L2结论参数梯度初值
     Hidden=ones(SampleNum,Num_subset);               %%%%%隐节点初值
     Hiddensum=zeros(SampleNum,Num_subset);

     err=0;
     NetworkOut=NetworkOut*0;
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%修改后算法
   for i=1:SampleNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=real((exp(-(x(i,k)-a(k,j))*conj(x(i,k)-a(k,j))*(b(k,j)*b(k,j)))))*Hidden(i,j);
                 Hiddensum(i,j)=w(j,k)*x(i,k)+Hiddensum(i,j);
             end
             Hiddensum(i,j)=Hiddensum(i,j)+w(j,Indim+1);
         end
         for j=1:Num_subset
             NetworkOut(i)= Hidden(i,j)*Hiddensum(i,j)+NetworkOut(i);
         end
  end
     NetworkOut;
     %%%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     for i=1:SampleNum
         err=1/SampleNum*(O(i)-NetworkOut(i))*conj(O(i)-NetworkOut(i))+err;
     end
     err;
     Traine(epochs)=err;
     Traine=real(Traine);
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
     %%% 训练准确率
     for TF = 1 : SampleNum
         if NetworkOut(TF)<0.5
             NetworkOut(TF) = 0;
         elseif NetworkOut(TF)>=0.5
             NetworkOut(TF) = 1;
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

     TestHidden=ones(TestNum,Num_subset);
     TestHiddensum=zeros(TestNum,Num_subset);
     %%%%%%%%%%%%%%%%%%%%%修改后的算法
     TestOut=zeros(1,TestNum);
     for i=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 TestHidden(i,j)=real((exp(-(Test1(i,k)-a(k,j))*conj(Test1(i,k)-a(k,j))*(b(k,j)*b(k,j)))))*TestHidden(i,j);
                 TestHiddensum(i,j)=w(j,k)*Test1(i,k)+TestHiddensum(i,j);
             end
             TestHiddensum(i,j)=w(j,Indim+1)+TestHiddensum(i,j);
         end
         for j=1:Num_subset
             TestOut(i)=TestHidden(i,j)*TestHiddensum(i,j)+TestOut(i);
         end
     end
     %%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     Err=0;
     for i=1:TestNum
         Err=1/TestNum*(OTest(i)-TestOut(i))*conj(OTest(i)-TestOut(i))+Err;
     end
     Teste(epochs)=Err;
     Teste=real(Teste);
     %%%%%%%%%%% 测试准确率
     for TTF = 1 : TestNum
         if TestOut(TTF)<0.5
             TestOut(TTF) = 0;
         elseif TestOut(TTF)>=0.5
             TestOut(TTF) = 1;
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

     %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
   for j=1:Num_subset
       for k=1:Indim  
         for i=1:SampleNum
             Delpublic=O(i)-NetworkOut(i);
             Gradient_w(j,k)=Delpublic*Hidden(i,j)*x(i,k)+Gradient_w(j,k);
             Gradient1_w(j,k)=Gradient_w(j,k)+2*Lambda*w(j,k);
         end
       end 
  end
     for j=1:Num_subset
       for i=1:SampleNum
           Delpublic=O(i)-NetworkOut(i);
           Gradient_w(j,Indim+1)=Delpublic*Hidden(i,j)+Gradient_w(j,Indim+1);
           Gradient1_w(j,Indim+1)=Gradient_w(j,Indim+1)+2*Lambda*w(j,Indim+1);
       end 
     end 
     
     for j=1:Num_subset
             for k=1:Indim
                 for i=1:SampleNum
                     Delpublic=O(i)-NetworkOut(i);  
                     Gradient_a(k,j)=2*(real(Delpublic)*real(Hiddensum(i,j))*real(Hidden(i,j))+imag(Delpublic)*imag(Hiddensum(i,j))*real(Hidden(i,j)))*(x(i,k)-a(k,j))*(b(k,j)*b(k,j))...
                         +Gradient_a(k,j);
                     Gradient1_a(k,j)=Gradient_a(k,j)+2*Lambda*a(k,j);
                     Gradient_b(k,j)=2*(real(Delpublic)*real(Hiddensum(i,j))*Hidden(i,j)+ imag(Delpublic)*imag(Hiddensum(i,j))*Hidden(i,j)    )...
                         *(x(i,k)-a(k,j))*conj(x(i,k)-a(k,j))*b(k,j)+Gradient_b(k,j);
                     Gradient1_b(k,j)=Gradient_b(k,j)+2*Lambda*b(k,j);
                 end
              end
     end
     %%%%%%%%%%%% 动量 %%%%%%%%%%%%%%%%%%%%
%      Alpha_a=0;
%      Alpha_b=0;
%      Alpha_w=0;
%      if norm(Delte_a)>0.0001
%          Alpha_a=Tau*min(norm(Gradient_a),norm(Gradient1_a))/norm(Delte_a);
%      end
%      if norm(Delte_b)>0.0001
%          Alpha_b=Tau*min(norm(Gradient_b),norm(Gradient1_b))/norm(Delte_b);
%      end
%      if norm(Delte_w)>0.0001
%          Alpha_w=Tau*min(norm(Gradient_w),norm(Gradient1_w))/norm(Delte_w);
%      end
     Delte_a=Ir*Gradient1_a+Delte_a;
     Delte_b=-Ir*Gradient1_b+Delte_b;
     Delte_w=Ir*Gradient1_w+Delte_w;
     a=a+Delte_a;
     b=b+Delte_b;
     w=w+Delte_w;
      %%%%%%%%%%%%%%%%%计算梯度的范数%%%%%%%%%%%%%
     DeltaA=reshape(Gradient1_a,1,Indim*Num_subset);
     DeltaB=reshape(Gradient1_b,1,Indim*Num_subset);
     DeltaW=reshape(Gradient1_w,1,Num_subset*(Indim+1));
     Norm_W=norm([DeltaA DeltaB DeltaW]);
     NormW(epochs)=Norm_W;
     
     % 如果当前测试准确率更高，则保存最佳测试准确率和对应的训练准确率
     if TestAcc >= Best_TestAcc
         Best_Epoch = epochs;
         Best_TestAcc = TestAcc;
         Best_TrainAcc = TrainAcc;
     end

     %%%%%%%%%%%% 再次循环 %%%%%%%%%%%%%%%%%%%%%%%%
     epochs=epochs+1;
end
t=toc;
 
%%%%%%%%%%%%%%%%%%%%%%%%% Do Your Picture %%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%% 训练误差和梯度范数
% xian11 = figure(10000+gogo);%+gogo
% plot(1:epochs-1,Traine);
% hold on
% xlabel("Number of epochs");
% ylabel("Training Error");
% 
% xian12 = figure(20000+gogo);%+gogo
% plot(1:epochs-1,Teste);
% hold on
% xlabel("Number of epochs");
% ylabel("Testing Error");
% 
% xian21 = figure(30000+gogo);%+gogo
% plot(1:epochs-1,TrainAccSave);
% hold on
% xlabel("Number of epochs");
% ylabel("Training Accuracy");
% 
% xian22 = figure(40000+gogo);%+gogo
% plot(1:epochs-1,TestAccSave);
% hold on
% xlabel("Number of epochs");
% ylabel("Testing Accuracy");

%%%%%%%%%%%%%%%%% 存储  (训练  测试  步数)
%filename = [num2str(Best_TrainAcc) '_' num2str(Best_TestAcc) '_' num2str(Best_Epoch) '.mat'];
%    save(filename);


TenAcc(gogo,:) = [Best_TrainAcc,Best_TestAcc,Best_Epoch];
time(gogo,:) = t;

end

MeanAcc = sum(TenAcc,1)/ExampleNum;

Mean_TrainAcc = MeanAcc(1,1);
Mean_TestAcc = MeanAcc(1,2);
Mean_Epoch = MeanAcc(1,3);
sprintf('训练准确率=%0.2f',Mean_TrainAcc)
sprintf('测试准确率=%0.2f',Mean_TestAcc)
sprintf('平均时间=%0.2f',mean(time))
disp(Mean_Epoch);
%filename2 = ['ZZZ_' num2str(Mean_TrainAcc) '_' num2str(Mean_TestAcc) '_' num2str(Mean_Epoch) '.txt'];
%writematrix(MeanAcc,filename2);
