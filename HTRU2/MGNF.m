%%%%%%%%%%% 改编模糊推理系统实值MGNF 吴老师的那篇文章  Housing %%%%%%%%%%%%%%
function Out = MGNF()
clear
clc
close all

%%%% 读取数据集
[Data] = LoadData_HTRU2();
% load('Data.mat')

x = Data.TrSamIn;       % 训练集输入
O = Data.TrSamOut;      % 训练集输出
Test1 = Data.TeSamIn;   % 测试集输入
OTest = Data.TeSamOut;  % 测试集输出

%%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
Indim=size(x,2);                       %%%%特征个数
Outdim=1;                     %%%%输出样本维数
TraNum=size(x,1);                  %%%%训练样本个数
NetworkOut=zeros(1,TraNum);
TestNum=size(Test1,1);              % 测试样本个数

%%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=12;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.000000001;                                 %%%%%误差标准
MaxEpochs=3000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0003;                                 %%%%%%学习率
e=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数
Acc = zeros(MaxEpochs-1,1);
a=rand(Indim,Num_subset);              %%%%%高斯函数的初始中心值
b=rand(Indim,Num_subset);               %%%%%高斯函数的初始宽度值     
w=rand(1,Num_subset);                %%%%%结论参数的初始值
l=0;                                    %%%%%跳出开关
t=0;
NetworkOut=zeros(1,TraNum);
Hidden=ones(TraNum,Num_subset);         %%%%%隐节点初值
Delta_a=a.*0;                                  %%%%%中心梯度初值
Delta_b=b.*0;                                  %%%%%宽度梯度初值
Delta_w=w.*0;                               %%%%%%%%%结论参数梯度初值
tic;
%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<MaxEpochs&l==0)
   
    %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
     Delta_a=a.*0;                                  %%%%%中心梯度初值
     Delta_b=b.*0;                                  %%%%%宽度梯度初值
     Delta_w=w.*0;                               %%%%%%%%%结论参数梯度初值
     Hidden=ones(TraNum,Num_subset);
     err=0;
     NetworkOut=NetworkOut*0;
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%
     for i=1:TraNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=exp(-(x(i,k)-a(k,j))*(x(i,k)-a(k,j))*(b(k,j)*(b(k,j))))*Hidden(i,j);
             end
         end
         for j=1:Num_subset
             NetworkOut(i)= Hidden(i,j)*w(j)+NetworkOut(i);
         end
     end
     NetworkOut;
     %%%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     for i=1:TraNum
         err=1/TraNum*(O(i)-NetworkOut(i))^2+err;
     end
     err;
     e(epochs)=err;

    %%% 计算训练精度 %%%
    NetworkOut1 = NetworkOut;
    NetworkOut1(NetworkOut1<0.5)=0;
    NetworkOut1(NetworkOut1>=0.5)=1;
    rightnumber=0;
    for i=1:TraNum
        if O(i)==NetworkOut1(i)
            rightnumber=rightnumber+1;
        end
    end
    Acc(epochs)=rightnumber/TraNum*100;
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
    %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
     for j=1:Num_subset
         for i=1:TraNum
             Delpublic=O(i)-NetworkOut(i);
             Delta_w(j)=Delpublic*Hidden(i,j)+Delta_w(j);
         end
     end
     for j=1:Num_subset
             for k=1:Indim
                 for i=1:TraNum
                     Delpublic=O(i)-NetworkOut(i);  
                     Delta_a(k,j)=2*Delpublic*w(j)*Hidden(i,j)*(x(i,k)-a(k,j))*(b(k,j)*b(k,j))+Delta_a(k,j);
                     Delta_b(k,j)=2*Delpublic*w(j)*Hidden(i,j)*(x(i,k)-a(k,j))*(x(i,k)-a(k,j))*b(k,j)+Delta_b(k,j);
                 end
              end
      end
     a=a+Ir*Delta_a;
     b=b-Ir*Delta_b;
     w=w+Ir*Delta_w;
     %%%%%%%%%%%%%%%%%计算梯度的范数%%%%%%%%%%%%%
     DeltaA=reshape(Delta_a,1,Indim*Num_subset);
     DeltaB=reshape(Delta_b,1,Indim*Num_subset);
     DeltaW=reshape(Delta_w,1,Num_subset);
     Norm_W=norm([DeltaA DeltaB DeltaW]);
     NormW(epochs)=Norm_W;
     %%%%%%%%%%%% 再次循环 %%%%%%%%%%%%%%%%%%%%%%%%
     epochs=epochs+1;
 end
 t=toc;
 epochs;
 


% c=1:epochs-1;
% figure(1)
% c=1:epochs-1;
% loglog(c,real(e),'k')
% figure(2)
% loglog(c,NormW,'k')
% figure(3)
% hold on 
% plot(c,Acc)
% hold off

 %%%%%%%%%%%%%%%%%%%%%%%%%%%% 计算测试样本的值 %%%%%%%%%%%%%
Hidden=ones(TestNum,Num_subset);
Hiddensum=zeros(TestNum,Num_subset);    
TestOut=zeros(1,TestNum); 
     for i=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=(exp(-(Test1(i,k)-a(k,j))*(Test1(i,k)-a(k,j))*(b(k,j)*b(k,j))))*Hidden(i,j);
             end
         end
         for j=1:Num_subset
             TestOut(i)=Hidden(i,j)*w(j)+TestOut(i);
         end
     end
 Err=0;
for i=1:TestNum
    Err=1/TestNum*(OTest(i)-TestOut(i))^2+Err;
end 
TestOut;

 %预测正确率 测试 %%%%
 TestOut;
 TestOut(TestOut<0.5)=0;
 TestOut(TestOut>=0.5)=1;

rightnumber=0;
for i=1:TestNum
    if OTest(i)==TestOut(i)
        rightnumber=rightnumber+1;
    end
end
rightratio_test=rightnumber/TestNum*100;

 Out = [Norm_W,t,err,Err, Acc(end), rightratio_test]
end
