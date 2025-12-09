不同数据集的代码在相应以数据集名字命名的文件夹里；个别文件夹内方法不全，则找Others文件夹中的mainV1.m主程序，子程序分别是：SCVNFMV1、SCVNFMV1_NoL2、
SCVNFMV1_NoMomentum、SCVNFMV1_NoMomentumL2

数据结果保存在Results.xlsx中
图像绘制：plotresults.m




历史记录：
一审：
不同学习率eta和不同自适应动量系数tau结果
save('example2_EtaTauParameterResults_AdapMoment.mat')
save('example2_EtaTauParameterResults_StabMoment.mat')

对比SCVNFM与对应的实数版模型MGNF的训练时间，代码在2_Heart、2_Liver、2_BreastCancer这三个文件夹里



分类问题：
最终版：10次实验的平均值，70%作为训练集，30%作为测试集

4个二分类实验：
2_Heart、2_Liver这两个数据集代码分别在这两个文件夹里；
另两个数据集Sonar和Breast Cancer Wisconsin的代码主程序是mainV1.m，子程序分别是：SCVNFMV1、SCVNFMV1_NoL2、
SCVNFMV1_NoMomentum、SCVNFMV1_NoMomentumL2
			
3个多分类实验：
Seeds_3fen、Cleveland_5fen、Glass_6fen这三个数据集代码分别在这三个文件夹里


”无L2和动量实验结果“这个文件夹图4 曲线对比结果


main.m是主程序，n次k折交叉验证

SCVNFM是子程序，给定训练集和测试集模型训练和预测情况

SCVNFMV1是子程序，交叉验证，给定训练集和测试集模型训练和预测情况，输出最好的结果，程序里用的这个
Sonar参数初始化：
    a=0.1*rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
    b=0.5*rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
    w=0.1*rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值
其他数据集：
    a=0.5*rand(Indim,Num_subset);               %%%%%高斯函数的初始中心值
    b=0.5*rand(Indim,Num_subset);                %%%%%高斯函数的初始宽度值     
    w=0.5*rand(Num_subset,Indim+1);                     %%%%%结论参数的初始值


SCVNFMV2 子程序，选择70%作为训练集，30%作为测试集

membership-functions文件夹中包含将隶属度函数替换成三角形、钟形函数后，在Wind数据集上的实验程序。



