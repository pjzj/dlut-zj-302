
% Author: Jian Zhou
% School of Computer Science and Technology, Dalian University of Technology
% Date: 19 Mar, 2019
% E-mail:zhoujian@mail.dlut.edu.cn

%=======Test_Laplacian Score=====================
clc
clear all;
close all;
  
%  fea = importdata('winequality.mat');
%  fea = fea(:,1:13);
 xlsread 'zj.xlsx';
 fea = ans(:,1:6);
 options = [];
 options.NeighborMode = 'KNN';
 options.k = 10;
 options.WeightMode = 'HeatKernel';
 options.t = 1;
 W = constructW(fea,options);
 LaplacianScore = LaplacianScore(fea,W)
 [junk, index] = sort(-LaplacianScore);
 newfea = fea(:,index);
 
%========Test_DBSCAN=========================
clc
clear all;
close all;

xlsread 'zj.xlsx';
X = ans(:,[1,6]); %1:2
epsilon=0.5;
MinPts=10;
IDX=DBSCAN(X,epsilon,MinPts);
PlotClusterinResult(X, IDX);
%set (gca, 'XTick',[0:0.1:50]);
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);

%========Test_Cluster Performance===============
clc
clear all;
close all;

load iris.dat
data = iris(:,1:4);
target = iris(:,5);
Idx = kmeans(data,3);

k=3;
result=cell(1,k);
for i=1:k
    result{i}=(find(Idx(:,1)==i))';
end
[ FM,P,MSE,NMI] = performace(data, result,target )

%==========SpectralClustering==================

% clear all;
% close all;
% clc
% x=0:0.05:2*pi;
% c=cos(x);
% s=sin(x);
% ss=s(find(s.*(x>=0&x<=3)));
% cc=c(find(c.*(x>=1.5&x<=4.5)));
% xc=x(find(x.*(x>=1.5&x<=4.5)));
% xs=x(find(x.*(x>=0&x<=3)));
% s1=randn(size(ss))./10+ss;
% c1=randn(size(cc))./10+cc;
% X=[[xc;c1],[xs;s1]]';

%  load iris.dat;
%  X=iris(:,[3,4]);

 xlsread 'zj.xlsx';
 X = ans(:,[1,6]);

[n,D]=size(X);
knear=5;
ind=2;
[NI,sigma,dist2]=Knearest_4(X,knear,ind);
Graph={'Laplace','LLE','Euclidean'};
CKSym=GraphType(D,n,Graph{1},knear,NI,dist2);
[Grps, SingVals] = SpectralClusterings(CKSym,3);

indx1=find(Grps==1);
indx2=find(Grps==2);
indx3=find(Grps==3);
%indx4=find(Grps==4);

plot(X(indx1,1),X(indx1,2),'ro');
hold on;
plot(X(indx2,1),X(indx2,2),'bd');
hold on;
plot(X(indx3,1),X(indx3,2),'x');

plot(X(indx4,1),X(indx4,2),'o');
hold on;


x=linspace(-10.0,10.0);
y=1./(1.0+exp(-1.0*x));
plot(x,y)
 set(gca,'LooseInset',get(gca,'TightInset'))
 set(gca,'xlabel')
 
%================Drawing======================
load iris.dat
data = iris(:,[3,4,5]);
n = size(data,2);
for i=1:size(data,1)
    if data(i,n)==0
        plot(data(i,1),data(i,2),'o','MarkerSize',6,'color',[0 0.545 0.271]);
        hold on;
    elseif data(i,n)==1
         plot(data(i,1),data(i,2),'x','MarkerSize',6,'Color',[34/255 139/255 34/255]);
         hold on;
    elseif data(i,n)==3
         plot(data(i,1),data(i,2),'o','MarkerSize',6,'Color',[1 0 0]);
         hold on;
    elseif data(i,n)==2
         plot(data(i,1),data(i,2),'*','MarkerSize',6,'Color',[0,0,1]);
         hold on;
    elseif data(i,n)==4
         plot(data(i,1),data(i,2),'.','Color',[128/255,0,0]);
         hold on;
    elseif data(i,n)==5
         plot(data(i,1),data(i,2),'.','Color',[1 0 1]);
         hold on;
    elseif data(i,n)==6
         plot(data(i,1),data(i,2),'.','Color',[0 0.40784 0.5451]);
         hold on;
    elseif data(i,n)==7
         plot(data(i,1),data(i,2),'.','Color',[0.933 0.071 0.4627]);
         hold on;
    elseif data(i,n)==8
         plot(data(i,1),data(i,2),'.','Color',[218/255,165/255,32/255]);
         hold on;
    elseif data(i,n)==9
         plot(data(i,1),data(i,2),'.','Color',[124/255,252/255,0]);
         hold on;
    end
end

%============Accuracy of KNN========================
clc
clear all;
close all;

load iris.dat
iris=iris(randperm(size(iris,1)),:);
data=iris(1:100,:);
tdata=iris(101:150,:);
% xlsread 'zj.xlsx';
% fea = ans(randperm(size(ans,1)),:);
% data = fea(1:40,:);
% tdata = fea(41:end,:);
col=size(data,2);
label=data(:,col);
tlabel=tdata(:,col);
for i=1:col-1
    training=data(:,i);
    testing=tdata(:,i);
    target = knnclassify( testing, training, label, 3);
    disp(['Accuracy of', '  ', 'Feature', ' ',num2str(i),':','  ',num2str(sum(tlabel==target)/(size(tdata,1)))]);
end

%====================BP Neural Network=============================
clc
clear all;
close all;

xlsread 'zj.xlsx';
data = ans';

data=data(:,randperm(size(data,2)));

input = data([2,4,5],:);
output = data([7],:);

input_train = input(:,1:80);
output_train = output(:,1:80);

input_test = input(:,81:end);
output_test = output(:,81:end);

net = newff(input_train,output_train,8);
net = train(net,input_train,output_train);

%==============Spearman===============
k1 = corr(fea(:,[6,7]))
k2 = corr(fea(:,[6,7]),'type','Spearman')
k3 = corr(fea(:,[6,7]),'type','Kendall')


k1 = corr(fea(:,[1,8]))
k2 = corr(fea(:,[2,8]))
k3 = corr(fea(:,[3,8]))
k4 = corr(fea(:,[4,8]))
k5 = corr(fea(:,[5,8]))
k6 = corr(fea(:,[6,8]))

%============Heat Map=================
clc
clear all;
close all;
x = [0.433,0,0,0,0,0;0.500,0.404,0,0,0,0;0.616,0.580,0.470,0,0,0;0.552,0.522,0.565,0.312,0,0;0.351,0.359,0.457,0.200,0.350,0;0.645,0.645,0.704,0.577,0.501,0.552];
XVarNames = {'JIF','5-Year JIF','CiteScore','SJR','SNIP','h5-Index'};
plotmatrix(x,'FillStyle','nofill','XVarNames',XVarNames,'YVarNames',XVarNames,'TextColor','Auto','ColorBar','on');

%============legend===================
lgd = legend('Journal 1','Journal 2','Journal 3','Journal 4','Journal 5','Journal 6','Journal 7','Journal 8','Journal 9','Journal 10','Journal 11','Journal 12','orientation','vertical','location','eastoutside')


h1 = parallelcoords(a(1:4,:));
hold on;
h2 = parallelcoords(a(5:8,:));
hold on;
h3 = parallelcoords(a(9:12,:));
hold on;

 legend(h1,'Journal 1','Journal 2','Journal 3','Journal 4','orientation','horizontal','location','north'); 
 legend boxoff;
 ah=axes('position',get(gca,'position'),'visible','off');
 legend(ah, h2,'Journal 5','Journal 6','Journal 7','Journal 8','orientation','horizontal','location','north');
 legend boxoff;
 ah=axes('position',get(gca,'position'),'visible','off');
 legend(ah,h3,'Journal 9','Journal 10','Journal 11','Journal 12','orientation','horizontal','location','north');
 legend boxoff;
 
 
 

