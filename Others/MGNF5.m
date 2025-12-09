%%%%%%% 改编模糊推理系统实值MGNF, 1/(2*pi*0.25)*exp(-((x1).^2+(x2).^2)/0.5).*cos(2*pi*(x1+x2)) %%%%%%%%%%%%%%
clear up
clc
%%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
Indim=2;                      %%%%输入样本维数
Outdim=1;                     %%%%输出样本维数
Num=5;
SampleNum=Num^2;          % 训练样本数量
Sample=-0.5:1/(Num-1):0.5;
[X1,X2]=ndgrid(Sample);   % 生成多变量函数的自变量序列
x1=X1(:);x2=X2(:);    %将矩阵按列顺序转化为向量
O=1/(2*pi*0.25)*exp(-((x1).^2+(x2).^2)/0.5).*cos(2*pi*(x1+x2));
x=[x1,x2];
NetworkOut=zeros(1,SampleNum);


Number=16;
TestNum=Number^2;           %测试样本数量
Sample=-0.5:1/(Number-1):0.5;
[X1,X2]=ndgrid(Sample);   % 生成多变量函数的自变量序列
x1=X1(:);x2=X2(:);    %将矩阵按列顺序转化为向量
OTest=1/(2*pi*0.25)*exp(-((x1).^2+(x2).^2)/0.5).*cos(2*pi*(x1+x2));
Test1=[x1,x2];
%%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=12;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.000000001;                                 %%%%%误差标准
MaxEpochs=15000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.011;                                 %%%%%%学习率
e=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数
a=rand(Indim,Num_subset);              %%%%%高斯函数的初始中心值
b=rand(Indim,Num_subset);               %%%%%高斯函数的初始宽度值     
w=rand(1,Num_subset);                %%%%%结论参数的初始值
l=0;                                    %%%%%跳出开关
t=0;
NetworkOut=zeros(1,SampleNum);
Hidden=ones(SampleNum,Num_subset);         %%%%%隐节点初值
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
     Hidden=ones(SampleNum,Num_subset);
     err=0;
     NetworkOut=NetworkOut*0;
     %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%
     for i=1:SampleNum
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
     for i=1:SampleNum
         err=1/SampleNum*(O(i)-NetworkOut(i))^2+err;
     end
     err;
     e(epochs)=err;
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
    %%%%%%%%%%% 计算梯度 更新权值%%%%%%%%%%%%%%%%%
     for j=1:Num_subset
         for i=1:SampleNum
             Delpublic=O(i)-NetworkOut(i);
             Delta_w(j)=Delpublic*Hidden(i,j)+Delta_w(j);
         end
     end
     for j=1:Num_subset
             for k=1:Indim
                 for i=1:SampleNum
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
 t
 epochs;
 

 err
%  c=1:epochs-1;
%   figure(1)
%   c=1:epochs-1;
% loglog(c,real(e),'k')
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
Err

