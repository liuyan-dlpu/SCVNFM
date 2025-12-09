%% SCVNFM结果
function results = SCVNFM(x, O, Test1, OTest) % X 是特征矩阵，y 是标签向量
[SampleNum, Indim] = size(x);       % the number of all samples
TestNum = length(OTest);       % the number of all samples

 %%%%%%%%%%% 设定参量 %%%%%%%%%%%%%%%%%%%%%%%%
Num_subset=8;             %%%%%%%%各语言变量对应的模糊子集的个数
HiddenUnit=Num_subset;                 %%%%隐节点数
err=0;                                  %%%%初始误差
E0=0.01;                                 %%%%%误差标准
MaxEpochs=10000;                          %%%%%最大迭代步数
epochs=1;                               %%%%%步数
Ir=0.0005;                                %%%%%%学习率
Tau=0.0004;                            %%%%动量常系数
Lambda=0.0004;                           %%%%正则项系数
e=[0,0];                                %%%%%存储误差
NormW=[0,0];                            %%%%%%存储梯度范数
a=rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
b=rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
w=rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值

l=0;                                    %%%%%跳出开关
Delte_a=a.*0;                                  %%%%%中心梯度初值
Delte_b=b.*0;                                  %%%%%宽度梯度初值
Delte_w=w.*0;                               %%%%%%%%%结论参数梯度初值
t=0;
tic;

NetworkOut=zeros(1,SampleNum);
%%%%%%%%%%%%%%%%%%%%%%  循环  %%%%%%%%%%%%%%%%%%%%
while(epochs<MaxEpochs&l==0)
   
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
     e(epochs)=err;
     e=real(e);
%      plot(1:MaxEpochs-1,e);

    %%%% 计算准确度 %%%%
     NetworkOut(find(NetworkOut<0.25))=0;
     NetworkOut(find(NetworkOut>=0.25&NetworkOut<0.75))=0.5;
     NetworkOut(find(NetworkOut>=0.75))=1;

    rightnumber=0;
    for i=1:SampleNum
        if O(i)==NetworkOut(i)
            rightnumber=rightnumber+1;
        end
    end
    rightratiotrain=rightnumber/SampleNum*100;
  
    
    
     %%%%%%%%%%%%%%%%%%%  判断   %%%%%%%%%%%%%%%%
     if(err<E0)
        l=1;
     end
     

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
     Alpha_a=0;
     Alpha_b=0;
     Alpha_w=0;
     if norm(Delte_a)>0.0001
         Alpha_a=Tau*min(norm(Gradient_a),norm(Gradient1_a))/norm(Delte_a);
     end
     if norm(Delte_b)>0.0001
         Alpha_b=Tau*min(norm(Gradient_b),norm(Gradient1_b))/norm(Delte_b);
     end
     if norm(Delte_w)>0.0001
         Alpha_w=Tau*min(norm(Gradient_w),norm(Gradient1_w))/norm(Delte_w);
     end
     Delte_a=Ir*Gradient_a+Alpha_a*Delte_a;
     Delte_b=-Ir*Gradient_b+Alpha_b*Delte_b;
     Delte_w=Ir*Gradient_w+Alpha_w*Delte_w;
     a=a+Delte_a;
     b=b+Delte_b;
     w=w+Delte_w;
      %%%%%%%%%%%%%%%%%计算梯度的范数%%%%%%%%%%%%%
     DeltaA=reshape(Gradient_a,1,Indim*Num_subset);
     DeltaB=reshape(Gradient_b,1,Indim*Num_subset);
     DeltaW=reshape(Gradient_w,1,Num_subset*(Indim+1));
     Norm_W=norm([DeltaA DeltaB DeltaW]);
     NormW(epochs)=Norm_W;
     
     %%%%%%%%%%%% 再次循环 %%%%%%%%%%%%%%%%%%%%%%%%
     epochs=epochs+1;
     
     
     
     
end

 t=toc;
 epochs;
 
 

c=1:epochs-1;
 

Hidden=ones(TestNum,Num_subset);
Hiddensum=zeros(TestNum,Num_subset);
  %%%%%%%%%%%%%%%%%%%%%修改后的算法
     TestOut=zeros(1,TestNum); 
     for i=1:TestNum
         for j=1:Num_subset
             for k=1:Indim
                 Hidden(i,j)=real((exp(-(Test1(i,k)-a(k,j))*conj(Test1(i,k)-a(k,j))*(b(k,j)*b(k,j)))))*Hidden(i,j);
                 Hiddensum(i,j)=w(j,k)*Test1(i,k)+Hiddensum(i,j);
             end
             Hiddensum(i,j)=w(j,Indim+1)+Hiddensum(i,j);
         end
         for j=1:Num_subset
             TestOut(i)=Hidden(i,j)*Hiddensum(i,j)+TestOut(i);
         end
     end
     %%%%%%%%%%%%%%%%%% 计算误差 %%%%%%%%%%%%%%
     Err=0;
     for i=1:TestNum
         Err=1/TestNum*(OTest(i)-TestOut(i))*conj(OTest(i)-TestOut(i))+Err;
     end
       err=real(err);
     Err=real(Err);
%       NetworkOut(find(NetworkOut<0.5))=0;
%      NetworkOut(find(NetworkOut>=0.5))=1;
     



     TestOut;
%      TestOut(find(TestOut<0.5))=0;
%      TestOut(find(TestOut>=0.5))=1;
     
     TestOut(find(TestOut<0.25))=0;
     TestOut(find(TestOut>=0.25&TestOut<0.75))=0.5;
     TestOut(find(TestOut>=0.75))=1;
 
 
%      TestOut(find(TestOut~=0&TestOut~=2))=1;
     %预测正确率
rightnumber=0;
for i=1:TestNum
    if OTest(i)==TestOut(i)
        rightnumber=rightnumber+1;
    end
end
rightratioTest=rightnumber/TestNum*100;
sprintf('测试准确率=%0.2f',rightratioTest);

results = [rightratiotrain, rightratioTest];
end

