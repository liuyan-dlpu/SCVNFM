%% 画图 拟合结果曲线对比图 训练Wind Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Results.xlsx', 'realworld', 'A3:SK6');
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
leg = legend('Original function','MGNF','ANCFIMM','SCVNFM','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TrainApproxResult_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TrainApproxResult_Wind.eps','-depsc2','-r600')

%% 画图 拟合结果曲线对比图 测试 Wind_Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Results.xlsx', 'realworld', 'B9:FM12');
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
leg = legend('Original function','MGNF','ANCFIMM','SCVNFM','FontSize',9);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.2 0.8 0.5]);
axis([0,xnum,-1.2,1.9]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TestApproxResult_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TestApproxResult_Wind.eps','-depsc2','-r600')

%% 画图 训练误差曲线图 Wind Speed%%
close all
ErrAll = xlsread('D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Results.xlsx','realworld','B15:NTP17');
figure; % 显式地创建一个新图形窗口，这是一个好习惯
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
ylabel('$\log_{10}$(MSE)','FontSize',18, 'Interpreter','Latex');
leg = legend('MGNF','ANCFIMM','SCVNFM','FontSize',14);
leg.ItemTokenSize = [38,40];
% set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
% set(get(gca,'YLabel'),'FontSize',14)
box on
set(gca,'Position',[0.15 0.15 0.8 0.8]);
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TrainErr_Wind.png','-dpng','-r600')
print(Fig,'D:\Documents\Matlab\L2 Momentum\V2split_1order_momentum_L2\Wind\TrainErr_Wind.eps','-depsc2','-r600')

