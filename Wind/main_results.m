%% N次运行的平均值 六个回归例子 %%
clear
addpath(genpath('D:\Documents\Matlab\full_higher_order\SixExamples'))
N  = 10;
Res = zeros(N+1,4);
for i = 1:N
%     Out = MGNF1();
%     Out = fully_1order1();
%     Out = CNFIS1();
%     Out = ANCFIMM();
    Out = FCNFA1();
    Res(i,:) = Out;
end
Res(N+1,:) = mean(Res,1);


%% N次运行的平均值 回归%%
clear
addpath(genpath('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill'))
N  = 10;
Res = zeros(N,4);
for i = 1:N
   Out = MGNF();
%    Out = fully_1order();
%    Out = ANCFIMM();
    Res(i,:) = Out;
end
mean(Res,1)

%% N次运行的平均值 分类%%
clear
addpath(genpath('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Shill'))
N  = 20;
Res = zeros(N,6);
for i = 1:N
    Out = MGNF();
%    Out = fully_1order();
%    Out = ANCFIMM();
    Res(i,:) = Out;
end
mean(Res,1)



%% 读取Housing.fig文件 师兄文中图13b Housing测试结果曲线图对比
clear 
close all
open('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\results figure\results figure\TrainingResult_Housing2.fig');
lh = findall(gca, 'type', 'line');% 如果图中有多条曲线，lh为一个数组
xc = get(lh, 'xdata');            % 取出x轴数据，xc是一个元胞数组
yc = get(lh, 'ydata');            % 取出y轴数据，yc是一个元胞数组

% 获取坐标
x=xc{1}; % x坐标
y1=yc{1}; % original function 曲线y坐标
y2=yc{2}; % Feedforward Type-1 FNN 曲线y坐标
y3=yc{3}; % Feedforward Type-2 FNN 曲线y坐标
y4=yc{4}; % NFNN-FCMnet-MCGA 曲线y坐标

%% 读取.mat文件
load('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\results figure\results figure\result_AllData_Housing2.mat')

plot(1:101,TeSamOut)


%% 保存数据
% Wind
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\AFCFIS_1Result0.0219.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\MGNF_1Result_0.0538.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\ANCFIMM_1Result_0.0551.mat')

% Liver
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\AFCFIS_1Result_73.8589.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\MGNF_1Result_70.9544.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\ANCFIMM_1Result_72.6141.mat')

% Iris
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\AFCFIS_1Result_99.0476.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\MGNF_1Result_97.7778.mat')
save('D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\ANCFIMM_1Result_98.0952.mat')

%% 画图 训练误差曲线图 Housing%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Housing', 'B105:NTP107');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Error','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TrainErr_Housing.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TrainErr_Housing.eps','-depsc2','-r600')

%% 画图 训练梯度的范数曲线图 Housing%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Housing', 'B110:NTP112');
hold on 
[AlgoNum, xnum] = size(ErrAll);
% plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
% plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
% plot(1:xnum, tansig(ErrAll(3,:)),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.01],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Norm of gradient','FontSize',18)
leg = legend('AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\GradNorm_Housing.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\GradNorm_Housing.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 训练Housing%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Housing', 'B93:OP96');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',1.5)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',1.5)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',1.5)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',1.5)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Training inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TrainApproxResult_Housing.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TrainApproxResult_Housing.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 训练Housing%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Housing', 'B99:CX102');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',1.5)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',1.5)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',1.5)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',1.5)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Test inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TestApproxResult_Housing.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Housing\TestApproxResult_Housing.eps','-depsc2','-r600')


%% 画图 训练误差曲线图 Wind Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Wind Speed', 'B54:NTP56');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Error','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TrainErr_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TrainErr_Wind.eps','-depsc2','-r600')

%% 画图 训练梯度的范数曲线图 Wind Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Wind Speed', 'B59:NTP61');
hold on 
[AlgoNum, xnum] = size(ErrAll);
% plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
% plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
% plot(1:xnum, tansig(ErrAll(3,:)),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.01],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Norm of gradient','FontSize',18)
leg = legend('AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\GradNorm_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\GradNorm_Wind.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 训练Wind Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Wind Speed', 'B42:SK45');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',1.5)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',1.5)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',1.5)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',1.5)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Training inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TrainApproxResult_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TrainApproxResult_Wind.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 测试 Wind_Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Wind Speed', 'B48:FM51');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',1.5)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',1.5)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',1.5)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',1.5)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Test inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TestApproxResult_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Wind\TestApproxResult_Wind.eps','-depsc2','-r600')


%% 画图 训练误差曲线图 Skill%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Skill', 'B36:DKJ38');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Error','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TrainErr_Skill.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TrainErr_Skill.eps','-depsc2','-r600')

%% 画图 训练梯度的范数曲线图 Skill%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Skill', 'B41:DKJ43');
hold on 
[AlgoNum, xnum] = size(ErrAll);
% plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
% plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
% plot(1:xnum, tansig(ErrAll(3,:)),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.01],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Norm of gradient','FontSize',18)
leg = legend('AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\GradNorm_Skill.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\GradNorm_Skill.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 训练Skill%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Skill', 'B24:CMK27');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',0.8)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',0.8)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',0.8)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',0.8)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Training inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TrainApproxResult_Skill.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TrainApproxResult_Skill.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 测试 Skill%%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Skill', 'B30:CMK33');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'-.','Color','#7FFF00','LineWidth',0.8)
plot(1:xnum, ErrAll(2,:),'Color','#191970','LineWidth',0.8)
plot(1:xnum, ErrAll(3,:),'--','Color','k','LineWidth',0.8)
plot(1:xnum, ErrAll(4,:),'-.','Color','r','LineWidth',0.8)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)%设置坐标轴刻度字体名称，大小
% title('Training approximation results','FontSize',14,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Test inputs','FontSize',13)
h = ylabel('$$F(x)$$','FontSize',13);
set(h,'Interpreter','latex');
leg = legend('Original function','MGNF','ANCFIMM','AFCFIS','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TestApproxResult_Skill.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Skill\TestApproxResult_Skill.eps','-depsc2','-r600')



%% 画图 训练误差曲线图 Liver %%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Liver', 'B48:DKJ50');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Error','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\TrainErr_Liver.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\TrainErr_Liver.eps','-depsc2','-r600')

%% 画图 训练精度曲线图 Liver %%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Liver', 'B43:DKJ45');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Accuracy (%)','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14, 'Location','southeast');
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\TrainAcc_Liver.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\TrainAcc_Liver.eps','-depsc2','-r600')

%% 画图 训练误差曲线图 Iris %%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Iris', 'B49:DKJ51');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Error','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\TrainErr_Iris.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\TrainErr_Iris.eps','-depsc2','-r600')

%% 画图 训练精度曲线图 Iris %%
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'Iris', 'B44:DKJ46');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#191970','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','k','LineWidth',2)
plot(1:xnum, ErrAll(3,:),'-.','Color','r','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',14)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',18)
ylabel('Accuracy (%)','FontSize',18)
leg = legend('MGNF','ANCFIMM','AFCFIS','FontSize',14, 'Location','southeast');
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
ylim([60,100])
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\TrainAcc_Iris.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\TrainAcc_Iris.eps','-depsc2','-r600')

%% Test accuracy results comparison on Liver%%
clear
close all

AlgNum = 4;
x = 1:AlgNum;
y = [69.13, 68.84,70.93,  71.15];
error = [5.02, 4.70, 2.74, 3.82];
neg = error;
pos = error;
hold on 
bar(y,0.5,'FaceColor',[0         0    0.5647]);
errorbar(x,y,neg, pos, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2)
xticks([1,2,3,4])
set(gca,'XTickLabel',{'MGNF','ANCFIMM', 'NFNN-FCMnet-MCGA','AFCFIS'},'FontSize',10)
hold on

set (gca,'position',[0.1,0.2,0.84,0.42] )
% axis([0.5,6.5,0,100]);
for i =1:AlgNum
    text(i+0.3,y(i)-2.5,num2str(y(i)),'FontSize',10)
end
xtickangle(15)
% xlim([0,5])
ylim([40,80])
xlabel('Algorithms','FontSize',12);
ylabel('Test accuracy (%)','FontSize',12);
box on
grid on
Fig = getimage(gca);  
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\LiverAccResutls.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Liver\LiverAccResutls.eps','-depsc2','-r600')


%% Test accuracy results comparison on Iris %%
clear
close all

AlgNum = 4;
x = 1:AlgNum;
y = [96.78, 96.89,97.03,  97.67];
error = [1.79,1.63, 2.69,  1.31];
neg = error;
pos = error;
hold on 
bar(y,0.5,'FaceColor',[0         0    0.5647]);
errorbar(x,y,neg, pos, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2)
xticks([1,2,3,4])
set(gca,'XTickLabel',{'MGNF','ANCFIMM', 'NFNN-FCMnet-MCGA','AFCFIS'},'FontSize',10)
hold on

set (gca,'position',[0.1,0.2,0.84,0.42] )
% axis([0.5,6.5,0,100]);
for i =1:AlgNum
    text(i+0.3,y(i)-2.5,num2str(y(i)),'FontSize',10)
end
xtickangle(15)
% xlim([0,5])
ylim([70,100])
xlabel('Algorithms','FontSize',12);
ylabel('Test accuracy (%)','FontSize',12);
box on
grid on
Fig = getimage(gca);  
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\IrisAccResutls.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\RealWorldDatasets\Iris\IrisAccResutls.eps','-depsc2','-r600')

%% 两个分类数据集上最后精度结果比较
clear;clc;close all;
% 获取到颜色
all_colors = {'#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295'};
% 生成示例数据
m = 2; % 两个数据集
n = 5; % 5种方法比较
x = 1:m;
y = [69.13, 71.28, 70.93, 68.56, 71.15;...
96.78, 97.03, 97.03, 96.89, 97.67];  % m*n
% 误差限
error = [5.02, 3.84, 2.74, 4.98, 3.82;...
1.79, 2.37,2.69, 1.63, 1.31];
neg = error;
pos = error;

% 多系列带有误差线的柱状图
figure;
% 绘制柱状图
h = bar(x, y);
% 设置每个系列颜色
for i = 1:length(h)
    h(1, i).FaceColor = all_colors{1,i};
end
% 单独设置第二个系列第二个柱子颜色
% % 这行代码少不了
% h(1, 2).FaceColor = 'flat';
% h(1, 2).CData(2,:) = all_colors{1,6}; 
% 获取误差线 x 值 
% 也就是 XEndPoints 的值
xx = zeros(m, n);
for i = 1 : n
    xx(:, i) = h(1, i).XEndPoints'; 
end
% 绘制误差线
hold on
errorbar(xx, y, neg, pos, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2);
hold off
% 绘制图例
legend({'MGNF', 'NFNN-FCMnet-GD', 'NFNN-FCMnet-MCGA','ANCFIMM','AFCFIS'}, 'Location','NorthEastOutside','FontSize',10);
% 设置 x 轴标签
set(gca, 'XTickLabel', {'Liver', 'Iris'},'FontSize',10);
xlabel('Datasets','FontSize',14)
ylabel('Accuracy (%)','FontSize',14)
set(gca,'Position',[0.1 0.15 0.6 0.35]);

%% Figure 8: Running Time %%
close all
t1=[0.05390,0.03286];
t2=[0.03926,0.03028];
y=[0.05390,0.03286;0.03926,0.03028;];
b=bar(y,0.65,'FaceColor','flat');

b(1).CData = [0         0    0.5647];
b(2).CData = [0.5059    0.0039         0];
% 绘制误差线
Error = [0.010182338	0.005876598;	0.0115318	0.00469016];
x = [0.85,1.15;1.85,2.15];
hold on
errorbar(x,y, Error, Error, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 1);
hold off
grid on
set(gca,'XTickLabel',{'Alpha-type','Triangular'},'FontSize',12)
legend('Function','Derivative');
xlabel('Spike response function','FontSize',14);
ylabel('Running time ({\it{ms}})','FontSize',14);
set (gca,'position',[0.11,0.2,0.84,0.6] )
axis([0.5,2.5,0,0.07]);
text(1-0.225,t1(1)+0.0021,num2str(t1(1)),'FontSize',10)
text(1+0.05,t1(2)+0.0021,num2str(t1(2)),'FontSize',10)
text(2-0.24,t2(1)+0.0021,num2str(t2(1)),'FontSize',10)
text(2+0.05,t2(2)+0.0021,num2str(t2(2)),'FontSize',10)
% saveas(gca,'TrainTime.eps')
Fig = getimage(gca);  
print(Fig,'RunTime','-dpng','-r600')

%% 读取.fig文件，保存里边的曲线数值
clear 
close all
% open('D:\Documents\Matlab\full_higher_order\SixExamples\fully_1order_error.fig');
open('D:\Documents\Matlab\full_higher_order\SixExamples\fully_1order_gradient.fig');
lh = findall(gca, 'type', 'line');% 如果图中有多条曲线，lh为一个数组
xc = get(lh, 'xdata');            % 取出x轴数据，xc是一个元胞数组
yc = get(lh, 'ydata');            % 取出y轴数据，yc是一个元胞数组

% 获取坐标
x=xc{1}; % x坐标
y1=yc{1}; % 获取第一条曲线y坐标
y2=yc{2}; 
y3=yc{3};
y4=yc{4}; 
y5=yc{5}; 
y6=yc{6}; 
hold on 
plot(x,y4)

%% 画图 图2 训练误差曲线图 AFCFIS在示例1-6上
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'SixExamsV1', 'B36:NTP41');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#385E0F','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','#0000FF','LineWidth',2)
plot(1:xnum, ErrAll(3,:),':','Color','#A020F0','LineWidth',2)
plot(1:xnum, ErrAll(4,:),'-.','Color','#B0171F','LineWidth',2)
plot(1:xnum, ErrAll(5,:),'Color','r','LineWidth',2)
plot(1:xnum, ErrAll(6,:),'--','Color','k','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',12)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',16)
ylabel('Error','FontSize',16)
leg = legend('Example 1','Example 2','Example 3','Example 4','Example 5','Example 6','FontSize',12);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\SixExamples\TrainErr_AFCFIS.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\SixExamples\TrainErr_AFCFIS.eps','-depsc2','-r600')


%% 画图 图3 梯度范数曲线图 AFCFIS在示例1-6上
close all
ErrAll = xlsread('D:\Documents\Matlab\full_higher_order\ExperimentResults.xlsx', 'SixExamsV1', 'B44:NTP49');
hold on 
[AlgoNum, xnum] = size(ErrAll);
plot(1:xnum, ErrAll(1,:),'Color','#385E0F','LineWidth',2)
plot(1:xnum, ErrAll(2,:),'--','Color','#0000FF','LineWidth',2)
plot(1:xnum, ErrAll(3,:),':','Color','#A020F0','LineWidth',2)
plot(1:xnum, ErrAll(4,:),'-.','Color','#B0171F','LineWidth',2)
plot(1:xnum, ErrAll(5,:),'Color','r','LineWidth',2)
plot(1:xnum, ErrAll(6,:),'--','Color','k','LineWidth',2)
hold off
set(gca,'FontName','Times New Roman','FontSize',12)%设置坐标轴刻度字体名称，大小
set(gca, 'YScale', 'log')
% title('(b) Housing','FontSize',17,'position',[5000 0.0025],'FontWeight','bold')
xlabel('Iteration','FontSize',16)
ylabel('Norm of gradient','FontSize',16)
leg = legend('Example 1','Example 2','Example 3','Example 4','Example 5','Example 6','FontSize',12);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\full_higher_order\SixExamples\NormGrad_AFCFIS.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\full_higher_order\SixExamples\NormGrad_AFCFIS.eps','-depsc2','-r600')
