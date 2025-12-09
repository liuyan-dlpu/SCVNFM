%%%%%%%%%% ANFIS Implementation (Stable, No NaN Version)
clear;clc;

ExampleNum = 10; 
Allresult = zeros(ExampleNum,4);

for gogo = 1:ExampleNum
    
    % ---------- Load Data ----------
    data = load("Glass.txt");
    InputNeuronsNum = size(data,2)-1;
    DataNum = size(data,1);

    % ---------- Determine Class Number ----------
    sorted_target = sort(data(:,end));
    label = sorted_target(1);
    j = 1;
    for i=2:DataNum
        if sorted_target(i) ~= label(j)
            j=j+1;
            label(j)=sorted_target(i);
        end
    end
    ClassNum = j;
    OutputNeuronsNum = ClassNum;

    % ---------- Shuffle ----------
    data = data(randperm(DataNum), :);

    % ---------- Split Data ----------
    TrainNum = round(0.7*DataNum);
    TestNum = DataNum - TrainNum;

    x = data(1:TrainNum,1:InputNeuronsNum);
    O = data(1:TrainNum,end);
    Test1 = data(TrainNum+1:end,1:InputNeuronsNum);
    OTest = data(TrainNum+1:end,end);

    % ---------- Parameters ----------
    Indim = InputNeuronsNum;
    Num_subset = 6;
    HiddenUnit = Num_subset;
    MaxEpochs = 1000;
    Ir = 0.02;

    % 防止宽度过小导致 exp 爆炸
    a = 0.1 * rand(Indim, Num_subset);
    b = 0.5 * rand(Indim, Num_subset) + 0.05;
    w = 0.1 * rand(Num_subset, Indim + 1);

    % ---------- Record ----------
    Traine = zeros(1, MaxEpochs);
    Teste = zeros(1, MaxEpochs);
    NormW = zeros(1, MaxEpochs);

    Best_TestAcc = 0; Best_TrainAcc=0; Best_Epoch = 0;

    tic;
    % ---------- Training Loop ----------
    for epoch = 1:MaxEpochs
        
        FireStrength = zeros(TrainNum, Num_subset);
        RuleOutput  = zeros(TrainNum, Num_subset);
        NetworkOut  = zeros(TrainNum,1);

        % ---------- Forward Pass ----------
        for i = 1:TrainNum
            for j = 1:Num_subset
                fire = 1;
                for k = 1:Indim
                    
                    diff = x(i,k)-a(k,j);
                    expo = -(diff^2)/(2*b(k,j)^2);
                    expo = max(min(expo,50),-50);   % 限制指数防爆炸
                    
                    mu = exp(expo);
                    fire = fire * mu;
                end
                FireStrength(i,j) = fire;

                out = w(j,end);
                for k = 1:Indim
                    out = out + w(j,k)*x(i,k);
                end
                RuleOutput(i,j) = out;
            end

            total_fire = sum(FireStrength(i,:)) + 1e-8;
            NF = FireStrength(i,:)/total_fire;

            NetworkOut(i) = sum(NF.*RuleOutput(i,:));
        end

        % ---------- Compute Training Error ----------
        err = mean((O - NetworkOut).^2);
        Traine(epoch) = err;

        % ---------- Classify ----------
        Nout_round = round(NetworkOut);
        Nout_round(Nout_round<1)=1;
        TrainAcc = sum(Nout_round==O)/TrainNum*100;

        % ---------- Test Phase ----------
        TestOut = zeros(TestNum,1);
        for i=1:TestNum
            Tfire = zeros(1,Num_subset);
            Tout  = zeros(1,Num_subset);
            for j=1:Num_subset
                fire=1;
                for k=1:Indim
                    diff=Test1(i,k)-a(k,j);
                    expo=-(diff^2)/(2*b(k,j)^2);
                    expo=max(min(expo,50),-50);
                    mu=exp(expo);
                    fire=fire*mu;
                end
                Tfire(j)=fire;
                out=w(j,end);
                for k=1:Indim
                    out=out+w(j,k)*Test1(i,k);
                end
                Tout(j)=out;
            end
            
            NF = Tfire/(sum(Tfire)+1e-8);
            TestOut(i)=sum(NF.*Tout);
        end
        
        TestErr = mean((OTest - TestOut).^2);
        Teste(epoch) = TestErr;

        Test_round = round(TestOut);
        Test_round(Test_round<1)=1;
        TestAcc = sum(Test_round==OTest)/TestNum*100;

        % ---------- Record Best ----------
        if TestAcc > Best_TestAcc
            Best_TestAcc = TestAcc;
            Best_TrainAcc = TrainAcc;
            Best_Epoch = epoch;
        end

        % ---------- Backpropagation ----------
        Gradient_a = zeros(size(a));
        Gradient_b = zeros(size(b));
        Gradient_w = zeros(size(w));

        for i = 1:TrainNum
            total_fire=sum(FireStrength(i,:))+1e-8;
            NF=FireStrength(i,:)/total_fire;
            err_term = (O(i)-NetworkOut(i));

            for j=1:Num_subset
                rule_diff = RuleOutput(i,j) - NetworkOut(i);
                for k=1:Indim

                    mu = FireStrength(i,j)^(1/Indim);
                    diff = x(i,k)-a(k,j);

                    % dw
                    Gradient_w(j,k) = Gradient_w(j,k) + err_term * NF(j) * x(i,k);
                    Gradient_w(j,end) = Gradient_w(j,end) + err_term * NF(j);

                    % da
                    da = err_term * rule_diff * NF(j)*(1-NF(j)) * diff/(b(k,j)^2);
                    if abs(da) > 1e6, da = sign(da)*1e6; end
                    Gradient_a(k,j) = Gradient_a(k,j) + da;

                    % db
                    db = err_term * rule_diff * NF(j)*(1-NF(j)) * diff^2/(b(k,j)^3);
                    if abs(db) > 1e6, db = sign(db)*1e6; end
                    Gradient_b(k,j) = Gradient_b(k,j) + db;
                end
            end
        end

        % ---------- Normalize Gradient ----------
        Gradient_a = Gradient_a/(1e-6 + norm(Gradient_a(:)));
        Gradient_b = Gradient_b/(1e-6 + norm(Gradient_b(:)));
        Gradient_w = Gradient_w/(1e-6 + norm(Gradient_w(:)));

        NormW(epoch)=norm([Gradient_a(:);Gradient_b(:);Gradient_w(:)]);

        % ---------- Update parameters ----------
        a = a + Ir*Gradient_a;
        b = b + Ir*Gradient_b;
        w = w + Ir*Gradient_w;

        % ---------- Prevent b <= 0 ----------
        b = max(b,0.05);

        % ---------- NaN Recovery ----------
        if any(isnan(a(:))) || any(isnan(b(:))) || any(isnan(w(:)))
            disp("⚠ NaN detected! Resetting parameters...");
            a = 0.1 * rand(Indim, Num_subset);
            b = 0.5 * rand(Indim, Num_subset) + 0.05;
            w = 0.1 * rand(Num_subset, Indim + 1);
        end

        % ---------- Display ----------
        if mod(epoch,100)==0
            fprintf("Epoch %d | TrainAcc=%.2f | TestAcc=%.2f | GradNorm=%.4f\n", ...
                epoch,TrainAcc,TestAcc,NormW(epoch));
        end
    end

    t = toc;
    Allresult(gogo,:)=[Best_TrainAcc,Best_TestAcc,Best_Epoch,t];
    fprintf("Best Test Acc=%.2f at epoch %d\n",Best_TestAcc,Best_Epoch);
end

%%%%%%%%%%% 计算平均结果 %%%%%%%%%%%%%%%%%%%%%%%%
Meanruselt = mean(Allresult, 1);
Mean_TrainAcc = Meanruselt(1);
Mean_TestAcc = Meanruselt(2);
Mean_Epoch = Meanruselt(3);
Mean_Time = Meanruselt(4);

fprintf('\n=== Final ANFIS Results ===\n');
fprintf('Average Training Accuracy: %.6f\n', Mean_TrainAcc);
fprintf('Average Testing Accuracy: %.6f\n', Mean_TestAcc);
fprintf('Average Epochs: %.0f\n', Mean_Epoch);
fprintf('Average Training Time: %.2f seconds\n', Mean_Time);



