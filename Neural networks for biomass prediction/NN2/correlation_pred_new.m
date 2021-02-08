clear all; close all; clc;

train_data=load('final_output_train.txt');
test_data=load('final_output_test.txt');


min_lim=min([min(train_data),min(test_data)]);
max_lim=max([max(train_data),max(test_data)]);
train_size=floor(size(train_data,1)/2);
test_size=floor(size(test_data,1)/2);
gt_train=train_data(1:train_size,:);
pred_train=train_data((train_size+1):(2*train_size),:);
gt_test=test_data(1:test_size,:);
pred_test=test_data((test_size+1):(2*test_size),:);

RMSE_train=sqrt(immse(gt_train,pred_train))
RMSE_test=sqrt(immse(gt_test,pred_test))

mdl = fitlm(pred_train(:),gt_train(:));
mdl2 = fitlm(pred_test(:),gt_test(:));

R2_distr=mdl.Rsquared.Ordinary;
R2_distr2=mdl2.Rsquared.Ordinary;

figure(1)
subplot(1,2,1)
scatter(gt_train(:),pred_train(:),5,'filled')
str1 = ['R^2 = ' num2str(R2_distr)];
text(0.25,0.85,str1,'FontSize',14)
axis square
box on
xlabel('simulations','FontSize', 15) 
ylabel('NN predictions','FontSize', 15)
title({'Training'; ['(',num2str(length(gt_train)),' points)']},'FontSize', 10)
xlim([min_lim,max_lim])
ylim([min_lim,max_lim])
set(gca,'FontSize',15)
set(gca,'LineWidth',2)

subplot(1,2,2)
scatter(gt_test(:),pred_test(:),5,'filled')
str2 = ['R^2 = ' num2str(R2_distr2)];
text(0.25,0.85,str2,'FontSize',14)
axis square
box on
xlabel('simulations','FontSize', 15) 
ylabel('NN predictions','FontSize', 15)
title({'Test'; ['(',num2str(length(gt_test)),' points)']},'FontSize', 10)
xlim([min_lim,max_lim])
ylim([min_lim,max_lim])

set(gca,'FontSize',15)
set(gca,'LineWidth',2)

