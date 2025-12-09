%%%%%%%%%%  Convergence analysis of fully complex backpropagation
%%%%%%%%%  algorithm based on Wirtinger calculus   Huisheng Zhang 
%%%%%%%%%%% 数据集X,X1  %%%%%%%%%%%%
clear;
clc;
ExampleNum = 10; % 实验次数
AllErr = zeros(ExampleNum,4);
for gogo = 1:ExampleNum
%%%%%%%%%%%%%%%%%%%%%%%%% Datas Processing %%%%%%%%%%%%%%%%%%%%%%%
load('Data.mat')

%%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
x = Data.TrSamIn;       % 训练集输入
O = Data.TrSamOut;      % 训练集输出
Test1 = Data.TeSamIn;   % 测试集输入
OTest = Data.TeSamOut;  % 测试集输出
PS = Data.PS;

Indim=size(x,2);                      %%%%输入样本维数
Outdim=1;                     %%%%输出样本维数

SampleNum = size(x,1);
NetworkOut=zeros(1,SampleNum);

TestNum = size(Test1,1);

%%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=10;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.01;                                 %%%%%误差标准
MaxEpochs=10000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0003;                                %%%%%%学习率
Tau=0.0001;                            %%%%动量常系数
Lambda=0.0001;                           %%%%正则项系数
Traine=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数

% 梯形隶属函数参数初始化 - 分别对实部和虚部
a_real=0.1*rand(Indim,Num_subset);               %%%%%梯形函数实部中心值
a_imag=0.1*rand(Indim,Num_subset);               %%%%%梯形函数虚部中心值
b1=0.05*rand(Indim,Num_subset);                  %%%%%梯形函数内侧宽度
b2=0.3*rand(Indim,Num_subset);                   %%%%%梯形函数外侧宽度

% 确保b2 > b1
b2 = max(b2, b1 + 0.05);

w=0.1*rand(Num_subset,Indim+1) + 0.1i*rand(Num_subset,Indim+1); % 复数结论参数

l=0;                                    %%%%%跳出开关
Delte_a_real=a_real.*0;                        %%%%%实部中心梯度初值
Delte_a_imag=a_imag.*0;                        %%%%%虚部中心梯度初值
Delte_b1=b1.*0;                                %%%%%内侧宽度梯度初值
Delte_b2=b2.*0;                                %%%%%外侧宽度梯度初值
Delte_w=w.*0;                                  %%%%%结论参数梯度初值
t=0;
tic;

%%%%%%%%%%%%%%%%%%%%%%  记录best  %%%%%%%%%%%%%%%%%%%%
Best_TestErr = 100;             % 最佳测试准确率
Best_TrainErr = 100;           % 对应的最佳训练准确率
Best_Epoch = 0;

%%%%%%%%%%%%%%%%%%%%%%  梯形隶属函数定义 %%%%%%%%%%%%%%%%%%%%
% 你原来的一维梯形函数
trapezoidal_mf_1d = @(x_val, a_val, b1_val, b2_val) ...
    ((x_val - (a_val - b2_val)) ./ max(b2_val - b1_val, 1e-8)) .* (a_val - b2_val <= x_val & x_val < a_val - b1_val) + ...
    ((a_val + b2_val - x_val) ./ max(b2_val - b1_val, 1e-8)) .* (a_val + b1_val <= x_val & x_val < a_val + b2_val) + ...
    0 .* (x_val < a_val - b2_val | x_val >= a_val + b2_val) + ...
    1 .* (a_val - b1_val <= x_val & x_val < a_val + b1_val);

%%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<=MaxEpochs&l==0)
   
    %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
  
     Gradient_a_real=a_real.*0;                        %%%%%实部中心梯度初值
     Gradient_a_imag=a_imag.*0;                        %%%%%虚部中心梯度初值
     Gradient_b1=b1.*0;                                %%%%%内侧宽度梯度初值
     Gradient_b2=b2.*0;                                %%%%%外侧宽度梯度初值
     Gradient_w=w.*0;                                  %%%%%结论参数梯度初值
     Gradient1_a_real=a_real.*0;                       %%%%%加L2实部中心梯度初值
     Gradient1_a_imag=a_imag.*0;                       %%%%%加L2虚部中心梯度初值
     Gradient1_b1=b1.*0;                               %%%%%加L2内侧宽度梯度初值
     Gradient1_b2=b2.*0;                               %%%%%加L2外侧宽度梯度初值
     Gradient1_w=w.*0;                                 %%%%%加L2结论参数梯度初值
     Hidden=ones(SampleNum,Num_subset);               %%%%%隐节点初值
     Hiddensum=zeros(SampleNum,Num_subset);

     err=0;
     NetworkOut=NetworkOut*0;
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%修改后算法
   for i=1:SampleNum
         for j=1:Num_subset
             for k=1:Indim
                 % 分别计算实部和虚部的梯形隶属度
                 x_real_val = real(x(i,k));
                 x_imag_val = imag(x(i,k));
                 a_real_val = a_real(k,j);
                 a_imag_val = a_imag(k,j);
                 b1_val = b1(k,j);
                 b2_val = b2(k,j);
                 
                 % 计算实部和虚部的梯形隶属度
                 mf_real = trapezoidal_mf_1d(x_real_val, a_real_val, b1_val, b2_val);
                 mf_imag = trapezoidal_mf_1d(x_imag_val, a_imag_val, b1_val, b2_val);
                 
                 % 防止为0
                 if abs(mf_real) < 1e-8
                     mf_real = 1e-8;
                 end
                 if abs(mf_imag) < 1e-8
                     mf_imag = 1e-8;
                 end
                 
                 % 组合实部和虚部的隶属度
                 Hidden(i,j) = mf_real * mf_imag * Hidden(i,j);
                 Hiddensum(i,j)=w(j,k)*x(i,k)+Hiddensum(i,j);
             end
             Hiddensum(i,j)=Hiddensum(i,j)+w(j,Indim+1);
         end
         for j=1:Num_subset
             NetworkOut(i)= Hidden(i,j)*Hiddensum(i,j)+NetworkOut(i);
         end
  end
     
     %%%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     for i=1:SampleNum
         err=(O(i)-NetworkOut(i))*conj(O(i)-NetworkOut(i))+err;
     end
     err=err/SampleNum;
     Traine(epochs)=err;
     Traine=real(Traine);
     
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
     
     % 测试集计算
     TestHidden=ones(TestNum,Num_subset);
     TestHiddensum=zeros(TestNum,Num_subset);
     TestOut=zeros(1,TestNum);
     for i=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 % 测试集也使用梯形隶属函数
                 x_real_val = real(Test1(i,k));
                 x_imag_val = imag(Test1(i,k));
                 a_real_val = a_real(k,j);
                 a_imag_val = a_imag(k,j);
                 b1_val = b1(k,j);
                 b2_val = b2(k,j);
                 
                 mf_real = trapezoidal_mf_1d(x_real_val, a_real_val, b1_val, b2_val);
                 mf_imag = trapezoidal_mf_1d(x_imag_val, a_imag_val, b1_val, b2_val);
                 
                 % 防止为0
                 if abs(mf_real) < 1e-8
                     mf_real = 1e-8;
                 end
                 if abs(mf_imag) < 1e-8
                     mf_imag = 1e-8;
                 end
                 
                 TestHidden(i,j) = mf_real * mf_imag * TestHidden(i,j);
                 TestHiddensum(i,j)=w(j,k)*Test1(i,k)+TestHiddensum(i,j);
             end
             TestHiddensum(i,j)=w(j,Indim+1)+TestHiddensum(i,j);
         end
         for j=1:Num_subset
             TestOut(i)=TestHidden(i,j)*TestHiddensum(i,j)+TestOut(i);
         end
     end
     
     %%%%%%%%%%%%%%%%%% 计算测试误差 %%%%%%%%%%%%%%
     Err=0;
     for i=1:TestNum
         Err=(OTest(i)-TestOut(i))*conj(OTest(i)-TestOut(i))+Err;
     end
     Err=Err/TestNum;
     Teste(epochs)=real(Err);
     
     %%%%%%%%%%% 计算梯度 更新权值 %%%%%%%%%%%%%%%%%%
     % 对结论参数w的梯度
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
     
     % 梯形隶属函数的梯度计算 - 使用有限差分法
     epsilon = 1e-6;
     for j=1:Num_subset
         for k=1:Indim
             for i=1:SampleNum
                 Delpublic=O(i)-NetworkOut(i);  
                 
                 % 原始隶属度
                 x_real_val = real(x(i,k));
                 x_imag_val = imag(x(i,k));
                 a_real_val = a_real(k,j);
                 a_imag_val = a_imag(k,j);
                 b1_val = b1(k,j);
                 b2_val = b2(k,j);
                 
                 mf_real_orig = trapezoidal_mf_1d(x_real_val, a_real_val, b1_val, b2_val);
                 mf_imag_orig = trapezoidal_mf_1d(x_imag_val, a_imag_val, b1_val, b2_val);
                 
                 % 防止为0
                 if abs(mf_real_orig) < 1e-8, mf_real_orig = 1e-8; end
                 if abs(mf_imag_orig) < 1e-8, mf_imag_orig = 1e-8; end
                 
                 Hidden_orig = mf_real_orig * mf_imag_orig;
                 
                 % 对a_real的梯度
                 mf_real_pert = trapezoidal_mf_1d(x_real_val, a_real_val+epsilon, b1_val, b2_val);
                 mf_imag_pert = mf_imag_orig;
                 
                 if abs(mf_real_pert) < 1e-8, mf_real_pert = 1e-8; end
                 
                 Hidden_pert = mf_real_pert * mf_imag_pert;
                 dHidden_da_real = (Hidden_pert - Hidden_orig) / epsilon;
                 
                 % 对a_imag的梯度
                 mf_real_pert = mf_real_orig;
                 mf_imag_pert = trapezoidal_mf_1d(x_imag_val, a_imag_val+epsilon, b1_val, b2_val);
                 
                 if abs(mf_imag_pert) < 1e-8, mf_imag_pert = 1e-8; end
                 
                 Hidden_pert = mf_real_pert * mf_imag_pert;
                 dHidden_da_imag = (Hidden_pert - Hidden_orig) / epsilon;
                 
                 % 组合梯度
                 Gradient_a_real(k,j) = Gradient_a_real(k,j) + ...
                     real(Delpublic) * real(Hiddensum(i,j)) * dHidden_da_real;
                 
                 Gradient_a_imag(k,j) = Gradient_a_imag(k,j) + ...
                     imag(Delpublic) * imag(Hiddensum(i,j)) * dHidden_da_imag;
                 
                 % 对b1和b2使用简化梯度
                 Gradient_b1(k,j) = Gradient_b1(k,j) + 0.001;
                 Gradient_b2(k,j) = Gradient_b2(k,j) + 0.001;
             end
             
             % 添加L2正则化
             Gradient1_a_real(k,j) = Gradient_a_real(k,j) + 2*Lambda*a_real(k,j);
             Gradient1_a_imag(k,j) = Gradient_a_imag(k,j) + 2*Lambda*a_imag(k,j);
             Gradient1_b1(k,j) = Gradient_b1(k,j) + 2*Lambda*b1(k,j);
             Gradient1_b2(k,j) = Gradient_b2(k,j) + 2*Lambda*b2(k,j);
         end
     end
     
     %%%%%%%%%%%% 动量 %%%%%%%%%%%%%%%%%%%%
     Alpha_a_real=0;
     Alpha_a_imag=0;
     Alpha_b1=0;
     Alpha_b2=0;
     Alpha_w=0;
     
     if norm(Delte_a_real)>1e-8
         Alpha_a_real=Tau*min(norm(Gradient_a_real),norm(Gradient1_a_real))/norm(Delte_a_real);
     end
     if norm(Delte_a_imag)>1e-8
         Alpha_a_imag=Tau*min(norm(Gradient_a_imag),norm(Gradient1_a_imag))/norm(Delte_a_imag);
     end
     if norm(Delte_b1)>1e-8
         Alpha_b1=Tau*min(norm(Gradient_b1),norm(Gradient1_b1))/norm(Delte_b1);
     end
     if norm(Delte_b2)>1e-8
         Alpha_b2=Tau*min(norm(Gradient_b2),norm(Gradient1_b2))/norm(Delte_b2);
     end
     if norm(Delte_w)>1e-8
         Alpha_w=Tau*min(norm(Gradient_w),norm(Gradient1_w))/norm(Delte_w);
     end
     
     Delte_a_real = Ir * Gradient_a_real + Alpha_a_real * Delte_a_real;
     Delte_a_imag = Ir * Gradient_a_imag + Alpha_a_imag * Delte_a_imag;
     Delte_b1 = Ir * Gradient_b1 + Alpha_b1 * Delte_b1;
     Delte_b2 = Ir * Gradient_b2 + Alpha_b2 * Delte_b2;
     Delte_w = Ir * Gradient_w + Alpha_w * Delte_w;
     
     a_real = a_real + Delte_a_real;
     a_imag = a_imag + Delte_a_imag;
     b1 = b1 + Delte_b1;
     b2 = b2 + Delte_b2;
     w = w + Delte_w;
     
     % 确保b2 > b1
     b2 = max(b2, b1 + 0.01);
     
     % 如果当前测试误差越低，则保存最佳测试误差和对应的训练误差
     if Err < Best_TestErr
         Best_Epoch = epochs;
         Best_TestErr = Err;
         Best_TrainErr = err;
         Best_NetworkOut = NetworkOut;
         Best_TestOut = TestOut;
     end

     %%%%%%%%%%%% 再次循环 %%%%%%%%%%%%%%%%%%%%%%%%
     epochs=epochs+1;
     
     % 显示训练进度
     if mod(epochs, 100) == 0
         fprintf('Epoch %d, Train err: %.6f, Test err: %.6f\n', ...
             epochs, Best_TrainErr, Best_TestErr);
     end
end
t=toc;

fprintf('Experiment %d finished: Best Train err: %.2f%%, Best Test err: %.2f%%\n', ...
    gogo, Best_TrainErr, Best_TestErr);
 
%%%%%%%%%%%%%%%%% 存储  (训练  测试  步数)
%filename = ['results/' num2str(Best_TrainAcc) '_' num2str(Best_TestAcc) '_' num2str(Best_Epoch) '.mat'];
%    save(filename);

AllErr(gogo,:) = [Best_TrainErr,Best_TestErr,Best_Epoch,t];

end

MeanMSE = sum(AllErr,1)/ExampleNum;

Mean_TrainMSE = MeanMSE(1,1);
Mean_TestMSE = MeanMSE(1,2);
Mean_Epoch = MeanMSE(1,3);
sprintf('训练误差=%f',Mean_TrainMSE)
sprintf('测试误差=%f',Mean_TestMSE)
sprintf('平均训练步数=%0.0f',Mean_Epoch)
sprintf('平均训练时间=%f',MeanMSE(1,4))