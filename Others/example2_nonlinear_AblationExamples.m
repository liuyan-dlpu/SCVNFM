%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SCVNFM_HyperParamScan.m
%  Sensitivity and ablation study for adaptive momentum scheme
%  Based on nonlinear plant modeling problem z(t)=z(t-1)/(1+z(t-1)^2)+n^3(t)
%  Author: Fang Liu group (modified experimental version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
rng('default');

%% ======================== 数据准备 ==========================
load z.mat;           % 原始信号
for t=1:1001
    X(t)=2*((z(t)-min(z))/(max(z)-min(z)))-1;
end
x = X(1:1000);        % 训练输入
O = X(2:1001);        % 训练输出
SampleNum = length(x);

%% ======================== 实验配置 ==========================
eta_list  = [7e-6, 1e-5, 4e-5, 7e-5,1e-4,4e-4,7e-4, 1e-3];      % 学习率候选
tau_list  = [7e-6, 1e-5, 4e-5, 7e-5,1e-4,4e-4,7e-4, 1e-3];    % 动量上界候选
lambda_list = [1e-4];                % 正则项(可扩展)
num_repeat = 5;                      % 每组重复次数取平均
MaxEpochs = 1000;
E0 = 1e-3;                           % 目标误差

%% ======================== 结果存储 ==========================
Result = struct();
r_id = 1;

%% ======================== 主循环 ==========================
for lam = lambda_list
    for eta = eta_list
        for tau = tau_list
            fprintf('Running λ=%.1e, η=%.1e, τ=%.1e\n',lam,eta,tau);
            err_list = zeros(1,num_repeat);
            epoch_list = zeros(1,num_repeat);

            for rep = 1:num_repeat
                [final_err, epoch, curve] = train_once(x,O,eta,tau,lam,MaxEpochs,E0);
                err_list(rep) = final_err;
                epoch_list(rep) = epoch;
            end

            Result(r_id).lambda = lam;
            Result(r_id).eta = eta;
            Result(r_id).tau = tau;
            Result(r_id).mean_err = mean(err_list);
            Result(r_id).std_err  = std(err_list);
            Result(r_id).mean_epoch = mean(epoch_list);
            Result(r_id).std_epoch  = std(epoch_list);
            r_id = r_id + 1;
        end
    end
end

%% ======================== 输出结果表 ==========================
fprintf('\n================= Summary of Hyperparameter Sensitivity =================\n');
fprintf('  λ\t\t\tη\t\t\tτ\t\t\tMeanErr\t\t\tStdErr\t\t\tMeanEpoch\n');
fprintf('-------------------------------------------------------------------------------\n');
for i = 1:length(Result)
    fprintf('%1.2e\t%1.2e\t%1.2e\t%.8f\t%.8f\t%.2f\n', ...
        Result(i).lambda, Result(i).eta, Result(i).tau, ...
        Result(i).mean_err, Result(i).std_err, Result(i).mean_epoch);
end
fprintf('-------------------------------------------------------------------------------\n');


%% ======================== 可选: 可视化 ==========================
% 绘制 η-τ 热力图 (固定 λ)
lambda_fixed = lambda_list(1);
vals = [Result([Result.lambda]==lambda_fixed).mean_err];
etas = [Result([Result.lambda]==lambda_fixed).eta];
taus = [Result([Result.lambda]==lambda_fixed).tau];
etas_u = unique(etas); 
taus_u = unique(taus);
Z = nan(length(taus_u), length(etas_u));
for i=1:length(etas_u)
    for j=1:length(taus_u)
        idx = find(etas==etas_u(i) & taus==taus_u(j));
        if ~isempty(idx), Z(j,i)=vals(idx); end
    end
end

figure;
imagesc(log10(etas_u), log10(taus_u), Z); % 可用 log10 显示
set(gca,'YDir','normal'); 
c = colorbar;
% 设置colorbar刻度数值的字体大小为12
c.FontSize = 10;
xlabel('log_{10}(\eta)','FontSize', 15); ylabel('log_{10}(\tau)','FontSize', 15);
% title('Mean Error' ,'FontSize', 18);

% 使用红色渐变: 越小越红，越大越白
cmap = [linspace(1,1,256)', linspace(0,1,256)', linspace(0,1,256)']; % 白->红
colormap(cmap);

% 在热力图上标注损失值
for i = 1:length(etas_u)
    for j = 1:length(taus_u)
        % 确保该位置有误差值
        if ~isnan(Z(j,i))
            % 使用log10(误差)作为文本标签显示
            text(log10(etas_u(i)), log10(taus_u(j)), ...
                 sprintf('%.4f', Z(j,i)), 'Color', 'black', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle','FontSize', 10);
        end
    end
end
% 使用print函数保存为彩色EPS文件，分辨率设置为300 DPI
print('-depsc', '-r300','TauEtaHeatMap.eps');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ==================== 子函数: 单次训练 ==========================
function [final_err, epochs, e_curve] = train_once(x,O,Ir,Tau,Lambda,MaxEpochs,E0)

Indim = 1; Outdim = 1; Num_subset = 5;
SampleNum = length(x);

% 初始化参数
a = rand(Indim,Num_subset);
b = rand(Indim,Num_subset);
w = rand(Num_subset,Indim+1);
Delte_a = zeros(size(a)); Delte_b = zeros(size(b)); Delte_w = zeros(size(w));
e_curve = zeros(1,MaxEpochs);

for epochs = 1:MaxEpochs
    Hidden = ones(SampleNum,Num_subset);
    Hiddensum = zeros(SampleNum,Num_subset);
    NetworkOut = zeros(1,SampleNum);

    % ---------- 前向传播 ----------
    for i=1:SampleNum
        for j=1:Num_subset
            Hidden(i,j) = real(exp(-(x(i)-a(:,j))*conj(x(i)-a(:,j))*(b(:,j)*b(:,j))))*Hidden(i,j);
            Hiddensum(i,j) = w(j,1:Indim)*x(i) + w(j,Indim+1);
        end
        for j=1:Num_subset
            NetworkOut(i) = Hidden(i,j)*Hiddensum(i,j) + NetworkOut(i);
        end
    end

    % ---------- 误差计算 ----------
    err = mean((O - NetworkOut).*conj(O - NetworkOut));
    e_curve(epochs) = real(err);
    if err < E0
        break;
    end

    % ---------- 梯度计算 ----------
    Gradient_a = zeros(size(a)); Gradient_b = zeros(size(b)); Gradient_w = zeros(size(w));
    for j=1:Num_subset
        for k=1:Indim
            for i=1:SampleNum
                Delpublic = O(i)-NetworkOut(i);
                Gradient_w(j,k) = Gradient_w(j,k) + Delpublic*Hidden(i,j)*x(i);
            end
        end
        for i=1:SampleNum
            Delpublic = O(i)-NetworkOut(i);
            Gradient_w(j,Indim+1) = Gradient_w(j,Indim+1) + Delpublic*Hidden(i,j);
        end
    end

    for j=1:Num_subset
        for k=1:Indim
            for i=1:SampleNum
                Delpublic = O(i)-NetworkOut(i);
                Gradient_a(k,j) = Gradient_a(k,j) + ...
                    2*(real(Delpublic)*real(Hiddensum(i,j))*real(Hidden(i,j)) + ...
                       imag(Delpublic)*imag(Hiddensum(i,j))*real(Hidden(i,j))) * ...
                       (x(i)-a(k,j))*(b(k,j)*b(k,j));
                Gradient_b(k,j) = Gradient_b(k,j) + ...
                    2*(real(Delpublic)*real(Hiddensum(i,j))*Hidden(i,j) + ...
                       imag(Delpublic)*imag(Hiddensum(i,j))*Hidden(i,j)) * ...
                       (x(i)-a(k,j))*conj(x(i)-a(k,j))*b(k,j);
            end
        end
    end

    % ---------- L2 正则 ----------
    Gradient1_a = Gradient_a + 2*Lambda*a;
    Gradient1_b = Gradient_b + 2*Lambda*b;
    Gradient1_w = Gradient_w + 2*Lambda*w;

    % ---------- 自适应动量 ----------
    Alpha_a = 0; Alpha_b = 0; Alpha_w = 0;
    if norm(Delte_a)>1e-4
        Alpha_a = Tau * min(norm(Gradient_a),norm(Gradient1_a)) / norm(Delte_a);
    end
    if norm(Delte_b)>1e-4
        Alpha_b = Tau * min(norm(Gradient_b),norm(Gradient1_b)) / norm(Delte_b);
    end
    if norm(Delte_w)>1e-4
        Alpha_w = Tau * min(norm(Gradient_w),norm(Gradient1_w)) / norm(Delte_w);
    end

    % % ---------- 固定动量 ----------
    % Alpha_a = Tau;
    % Alpha_b = Tau;
    % Alpha_w = Tau;

    % ---------- 参数更新 ----------
    Delte_a = Ir*Gradient1_a + Alpha_a*Delte_a;
    Delte_b = -Ir*Gradient1_b + Alpha_b*Delte_b;
    Delte_w = Ir*Gradient1_w + Alpha_w*Delte_w;
    a = a + Delte_a; b = b + Delte_b; w = w + Delte_w;
end

final_err = real(err);
e_curve = e_curve(1:epochs);

end
