%%%%%%%%%%% split_1order，Housing  %%%%%%%%%%%%
% function Out = ANCFIMM()
clear up
clc
close all

%%%% 读取数据集
% [Data] = LoadData_Wind_finally();
load('Data.mat')

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
E0=0.00000;                                 %%%%%误差标准
MaxEpochs=10000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0015;                                 %%%%%%学习率
e=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数
a=rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
b=rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
w=rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值
l=0;                                    %%%%%跳出开关
Delta_a=a.*0;                                  %%%%%中心梯度初值
Delta_b=b.*0;                                  %%%%%宽度梯度初值
Delta_w=w.*0;                               %%%%%%%%%结论参数梯度初值
t=0;
tic;
%%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<MaxEpochs&l==0)
   
    %%%%%%%%%%%%%%%%%%%%% 重置参数 %%%%%%%%%%%%% 
     Delta_a=a.*0;                                  %%%%%中心梯度初值
     Delta_b=b.*0;                                  %%%%%宽度梯度初值
     Delta_w=w.*0;                               %%%%%%%%%结论参数梯度初值
     Hidden=ones(TraNum,Num_subset);
     Hiddensum=zeros(TraNum,Num_subset);

     err=0;
     NetworkOut=NetworkOut*0;
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%修改后算法
   for i=1:TraNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=(exp(-(x(i,k)-a(k,j))*conj(x(i,k)-a(k,j))*(b(k,j)*b(k,j))))*Hidden(i,j);
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
     for i=1:TraNum
         err=1/TraNum*(O(i)-NetworkOut(i))*conj(O(i)-NetworkOut(i))+err;
     end
     err;
     e(epochs)=err;
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
    %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
   for j=1:Num_subset
       for k=1:Indim  
         for i=1:TraNum
             Delpublic=O(i)-NetworkOut(i);
             Delta_w(j,k)=Delpublic*Hidden(i,j)*x(i,k)+Delta_w(j,k);
         end
       end 
   end
     for j=1:Num_subset
       for i=1:TraNum
           Delpublic=O(i)-NetworkOut(i);
           Delta_w(j,Indim+1)=Delpublic*Hidden(i,j)+Delta_w(j,Indim+1);
       end 
     end     
     for j=1:Num_subset
             for k=1:Indim
                 for i=1:TraNum
                     Delpublic=O(i)-NetworkOut(i);  
                     Delta_a(k,j)=2*(real(Delpublic)*real(Hiddensum(i,j))*real(Hidden(i,j))+imag(Delpublic)*imag(Hiddensum(i,j))*imag(Hidden(i,j)))*(x(i)-a(k,j))*(b(k,j)*b(k,j))+Delta_a(k,j);
                     Delta_b(k,j)=2*(real(Delpublic)*real(Hiddensum(i,j))*Hidden(i,j)+ imag(Delpublic)*imag(Hiddensum(i,j))*Hidden(i,j)    )...
                         *(x(i,k)-a(k,j))*conj(x(i,k)-a(k,j))*b(k,j)+Delta_b(k,j);
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
 
 figure
  c=1:epochs-1;
subplot(2,2,1);
 loglog(c,e,'k')
xlabel('number of iterations')
ylabel('error')
c=1:epochs-1;
subplot(2,2,2);  %%%%%%梯度范数变化曲线
loglog(c,NormW,'k')
xlabel('number of iterations')
ylabel('norm of gradient')
 

 c=1:epochs-1;
 

Hidden=ones(TestNum,Num_subset);
Hiddensum=zeros(TestNum,Num_subset);
  %%%%%%%%%%%%%%%%%%%%%修改后的算法
     TestOut=zeros(1,TestNum); 
     for i=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=(exp(-(Test1(i,k)-a(k,j))*conj(Test1(i,k)-a(k,j))*(b(k,j)*b(k,j))))*Hidden(i,j);
                 Hiddensum(i,j)=w(j,k)*Test1(i,k)+Hiddensum(i,j);
             end
             Hiddensum(i,j)=w(j,Indim+1)+Hiddensum(i,j);
         end
         for j=1:Num_subset
             TestOut(i)=Hidden(i,j)*Hiddensum(i,j)+TestOut(i);
         end
     end
     %%%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     Err=0;
     for i=1:TestNum
         Err=1/TestNum*(TestOut(i)-OTest(i))*conj(TestOut(i)-OTest(i))+Err;
     end
     err=real(err);
     Err=real(Err);
     TestOut;

 Out = [Norm_W,t,err,Err]
     
% end




