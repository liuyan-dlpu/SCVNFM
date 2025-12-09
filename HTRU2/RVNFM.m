ExampleNum = 10; % 实验次数
AllResults = zeros(ExampleNum,4);

for gogo = 1:ExampleNum
    %%%%%%%%%%%%%%%%%%%%%%%%% Datas Processing %%%%%%%%%%%%%%%%%%%%%%%
    % Breast Cancer Wisconsin
    [Data] = LoadData_HTRU2();
    % load('Data.mat')
    
    x = Data.TrSamIn;       % 训练集输入
    O = Data.TrSamOut;      % 训练集输出
    Test1 = Data.TeSamIn;   % 测试集输入
    OTest = Data.TeSamOut;  % 测试集输出             %   测试样本标签
    
    %%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
    
    Indim=size(x,2);                       %%%%特征个数
    
    TraNum=size(x,1);                  %%%%训练样本个数
    NetworkOut=zeros(1,TraNum);
    TestNum=size(Test1,1); 
    Outdim=1;                     %%%%输出样本维数
    
    SampleNum = TraNum;
    


    %%%%%%%%%%% 参数设定 %%%%%%%%%%%%%%%%%%%%%%%%
    Num_subset=12;             
    HiddenUnit=Num_subset;                 
    err=0;                                  
    E0=0.01;                                
    MaxEpochs=3000;                          
    epochs=1;                               
    Ir=0.0002;                                
    Tau=0.0001;                            
    Lambda=0.0001;                           
    Traine=[0,0];                                
    NormW=[0,0];                            
    a=0.1*rand(Indim,Num_subset);               
    b=0.5*rand(Indim,Num_subset);                
    w=0.1*rand(Num_subset,Indim+1);                     

    l=0;                                    
    Delte_a=a.*0;                                  
    Delte_b=b.*0;                                  
    Delte_w=w.*0;                               
    t=0;
    tic;

    %%%%%%%%%%%%%%%%%%%%%% 记录best  %%%%%%%%%%%%%%%%%%%%
    Best_TestAcc = 0;             
    Best_TrainAcc = 0;           
    Best_Epoch = 0;

    %%%%%%%%%%%%%%%%%%%%%% 循环  %%%%%%%%%%%%%%%%%%%%
    while(epochs<=MaxEpochs && l==0)
        %%%%%%%%%%%%%%%%%%%%% 重置梯度 %%%%%%%%%%%%% 
        Gradient_a=a.*0;                                  
        Gradient_b=b.*0;                                  
        Gradient_w=w.*0;                               
        Gradient1_a=a.*0;                                  
        Gradient1_b=b.*0;                                  
        Gradient1_w=w.*0;                               
        Hidden=ones(SampleNum,Num_subset);               
        Hiddensum=zeros(SampleNum,Num_subset);

        err=0;
        NetworkOut=NetworkOut*0;

        %%%%%%%%%%%%%%%%%%%%%计算输出值%%%%%%%%%
        for i=1:SampleNum
            for j=1:Num_subset
                for k=1:Indim
                    Hidden(i,j)=exp(-(x(i,k)-a(k,j))^2*b(k,j)^2)*Hidden(i,j);
                    Hiddensum(i,j)=w(j,k)*x(i,k)+Hiddensum(i,j);
                end
                Hiddensum(i,j)=Hiddensum(i,j)+w(j,Indim+1);
            end
            for j=1:Num_subset
                NetworkOut(i)= Hidden(i,j)*Hiddensum(i,j)+NetworkOut(i);
            end
        end

        %%%%%%%%%%%%%%%%%%%% 计算训练误差 %%%%%%%%%%%%%%
        for i=1:SampleNum
            err=1/SampleNum*(O(i)-NetworkOut(i))^2+err;
        end
        Traine(epochs)=err;

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
        TrainAccSave(epochs) = TrainAcc;

        %%%%%%%%%%% 测试输出 %%%%%%%%%%%%%%
        TestHidden=ones(TestNum,Num_subset);
        TestHiddensum=zeros(TestNum,Num_subset);
        TestOut=zeros(1,TestNum);

        for i=1:TestNum
            for j=1:Num_subset
                for k=1:Indim
                    TestHidden(i,j)=exp(-(Test1(i,k)-a(k,j))^2*b(k,j)^2)*TestHidden(i,j);
                    TestHiddensum(i,j)=w(j,k)*Test1(i,k)+TestHiddensum(i,j);
                end
                TestHiddensum(i,j)=w(j,Indim+1)+TestHiddensum(i,j);
            end
            for j=1:Num_subset
                TestOut(i)=TestHidden(i,j)*TestHiddensum(i,j)+TestOut(i);
            end
        end

        Err=0;
        for i=1:TestNum
            Err=1/TestNum*(OTest(i)-TestOut(i))^2+Err;
        end
        Teste(epochs)=Err;

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

        %%%%%%%%%%% 计算梯度并更新权值 %%%%%%%%%%%%%%
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
                    Gradient_a(k,j)=2*Delpublic*Hiddensum(i,j)*Hidden(i,j)*(x(i,k)-a(k,j))*(b(k,j)^2)+Gradient_a(k,j);
                    Gradient1_a(k,j)=Gradient_a(k,j)+2*Lambda*a(k,j);
                    Gradient_b(k,j)=2*Delpublic*Hiddensum(i,j)*Hidden(i,j)*(x(i,k)-a(k,j))^2*b(k,j)+Gradient_b(k,j);
                    Gradient1_b(k,j)=Gradient_b(k,j)+2*Lambda*b(k,j);
                end
            end
        end

        %%%%%%%%%%%% 动量 %%%%%%%%%%%%%%%%%%%%
        Alpha_a=0; Alpha_b=0; Alpha_w=0;
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

        DeltaA=reshape(Gradient_a,1,Indim*Num_subset);
        DeltaB=reshape(Gradient_b,1,Indim*Num_subset);
        DeltaW=reshape(Gradient_w,1,Num_subset*(Indim+1));
        Norm_W=norm([DeltaA DeltaB DeltaW]);
        NormW(epochs)=Norm_W;

        if TestAcc >= Best_TestAcc
            Best_Epoch = epochs;
            Best_TestAcc = TestAcc;
            Best_TrainAcc = TrainAcc;
        end

        epochs=epochs+1;
    end

    t=toc;
    %filename = ['results/' num2str(Best_TrainAcc) '_' num2str(Best_TestAcc) '_' num2str(Best_Epoch) '.mat'];
    %save(filename);

    AllResults(gogo,:) = [Best_TrainAcc,Best_TestAcc,Best_Epoch,t];

end

MeanResults = sum(AllResults,1)/ExampleNum;
Mean_TrainAcc = MeanResults(1,1);
Mean_TestAcc = MeanResults(1,2);
Mean_Epoch = MeanResults(1,3);
sprintf('训练准确率=%0.2f',Mean_TrainAcc)
sprintf('测试准确率=%0.2f',Mean_TestAcc)
sprintf('平均训练步数=%0.0f',Mean_Epoch)
sprintf('平均训练时间=%0.2f',MeanResults(1,4))
%filename2 = ['results/' 'ZZZ_' num2str(Mean_TrainAcc) '_' num2str(Mean_TestAcc) '_' num2str(Mean_Epoch) '.txt'];
%writematrix(MeanResults,filename2);
