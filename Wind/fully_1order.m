%% 用师兄的读取数据的代码，运用到自己的代码上 Housing
% function Out = fully_1order()
clear
clc
close all

%%%% 读取数据集
[Data] = LoadData_Wind_finally();
% [Data] = LoadData_Housing_Original();

x = Data.TrSamIn;       % 训练集输入
O = Data.TrSamOut;      % 训练集输出
Test1 = Data.TeSamIn;   % 测试集输入
OTest = Data.TeSamOut;  % 测试集输出
PS = Data.PS;

%%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
Indim=size(x,2);                       %%%%特征个数
Outdim=1;                     %%%%输出样本维数
TraNum=size(x,1);                  %%%%训练样本个数
NetworkOut=zeros(1,TraNum);
TestNum=size(Test1,1);              % 测试样本个数

%%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=10;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.000000001;                                 %%%%%误差标准
MaxEpochs=10000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0015;                                 %%%%%%学习率
e=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数 
a=rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
b=rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
w=rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值
l=0;                                    %%%%%跳出开关
t=0;                                    %时间
tic;
%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<MaxEpochs&l==0)
   
    %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
     Delta_a=a.*0;                                  %%%%%中心梯度初值
     Delta_b=b.*0;                                  %%%%%宽度梯度初值
     Delta_w=w.*0;                               %%%%%%%%%结论参数梯度初值
     Hidden=ones(TraNum,Num_subset);
     Hiddensum=zeros(TraNum,Num_subset);
     err=0;
     NetworkOut=NetworkOut*0;
     
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%
     for m=1:TraNum
         for j=1:Num_subset
             for k=1:Indim
                Hidden(m,j)=real(exp(-(x(m,k)-a(k,j))*conj(x(m,k)-a(k,j))*b(k,j)*conj(b(k,j))))*Hidden(m,j);
                Hiddensum(m,j)=w(j,k)*x(m,k)+Hiddensum(m,j);
             end
             Hiddensum(m,j)=Hiddensum(m,j)+w(j,Indim+1);
         end
         for j=1:Num_subset
             NetworkOut(m)= Hidden(m,j)*w(j)+NetworkOut(m);
             NetworkOut(m)= Hidden(m,j)*Hiddensum(m,j)+NetworkOut(m);
         end
     end
     NetworkOut;
     %%%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     for j=1:TraNum
         err=1/2*(O(j)-NetworkOut(j))*(conj(O(j)-NetworkOut(j)))+err;
     end
     err=1/TraNum*err;
     e(epochs)=err;
      %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
     %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
     for j=1:Num_subset
         for k=1:Indim
             for m=1:TraNum
                 Delpublic=O(m)-NetworkOut(m);
                 Delta_w(j,k)=1/2*(conj(Delpublic)*x(m,k)+Delpublic*conj(x(m,k)))* Hidden(m,j) +Delta_w(j,k);
             end
         end
     end
     for j=1:Num_subset
       for m=1:TraNum
           Delpublic=O(m)-NetworkOut(m);
           Delta_w(j,Indim+1)=real(Delpublic)*Hidden(m,j)+Delta_w(j,Indim+1);
       end 
     end  
     for j=1:Num_subset
             for k=1:Indim
                 for m=1:TraNum
                     Delpublic=O(m)-NetworkOut(m); 
                     Delta_a(k,j)=1/2*(conj(Delpublic)*Hiddensum(m,j)+Delpublic*conj(Hiddensum(m,j)))*Hidden(m,j)*(x(m,k)-a(k,j))*b(k,j)*conj(b(k,j))+Delta_a(k,j);
                     Delta_b(k,j)=1/2*(conj(Delpublic)*Hiddensum(m,j)+Delpublic*conj(Hiddensum(m,j)))*Hidden(m,j) ...
                         *(x(m,k)-a(k,j))*conj(x(m,k)-a(k,j))*b(k,j)+Delta_b(k,j);
                 end
             end
     end
     a=a+Ir*Delta_a;
     b=b-Ir*Delta_b;
     w=w+Ir*Delta_w;
     %%%%%%%%%%%%%%%%%计算梯度的范数%%%%%%%%%%%%%
     DeltaA=reshape(Delta_a,1,Indim*Num_subset);
     DeltaB=reshape(Delta_b,1,Indim*Num_subset);
     DeltaW=reshape(Delta_w,1,Num_subset*(Indim+1));
     Norm_W=norm([DeltaA DeltaB DeltaW]);
      NormW(epochs)=Norm_W;
     %%%%%%%%%%%% 再次循环 %%%%%%%%%%%%%%%%%%%%%%%%
     epochs=epochs+1;
end

 t=toc;
 epochs;
figure(1)
 hold on
  c=1:epochs-1;
loglog(c,real(e))
hold off
figure(2)
hold on 
loglog(c,NormW)
hold off


%%%%%%%%%%%%%%%%%%%%% 测试 %%%%%%%%%%
Hidden=ones(TestNum,Num_subset);
Hiddensum=zeros(TestNum,Num_subset);
     TestOut=zeros(1,TestNum); 
     for m=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(m,j)=real(exp(-(Test1(m,k)-a(k,j))*conj(Test1(m,k)-a(k,j))*b(k,j)*conj(b(k,j))))*Hidden(m,j);
                Hiddensum(m,j)=w(j,k)*Test1(m,k)+Hiddensum(m,j);
             end
             Hiddensum(m,j)=Hiddensum(m,j)+w(j,Indim+1);
         end
         for j=1:Num_subset
             TestOut(m)= Hidden(m,j)*w(j)+TestOut(m);
             TestOut(m)= Hidden(m,j)*Hiddensum(m,j)+TestOut(m);
         end
     end
     %%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     Err=0;
     for m=1:TestNum
         Err=1/TestNum*1/2*(OTest(m)-TestOut(m))*conj(OTest(m)-TestOut(m))+Err;
     end
     
     Err=real(Err);
     TestOut;


 Out = [Norm_W,t,err,Err]

% %%% 把训练曲线和测试曲线画出来 %%%
% figure (3)
% hold on
% plot(1:TraNum,O)
% plot(1:TraNum,NetworkOut)
% 
% figure (4)
% hold on
% plot(1:TestNum,OTest)
% plot(1:TestNum,TestOut)

% end
