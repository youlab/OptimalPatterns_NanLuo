clear all; close all; clc;

train_data=load('final_output_train_ensemble.txt');%ground truth, NN1, NN2, NN3,NN4,NN5,NN6,NN7, index of chosen NN, ensemble predictions, average disagreement between NNs,error between ensemble predictions and gt 
test_data=load('final_output_test_ensemble.txt');

train_size=size(train_data,1);
test_size=size(test_data,1);

all_data=[train_data;test_data];
all_data_size=size(all_data,1);

gt_all=all_data(:,1);
pred_all=all_data(:,6);
min_lim=min(min(all_data(:,1:4)));
max_lim=max(max(all_data(:,1:4)));


gt_train=gt_all(1:train_size);
pred_train=pred_all(1:train_size);
gt_test=gt_all((train_size+1):end);
pred_test=pred_all((train_size+1):end);

RMSE_train=sqrt(immse(gt_train,pred_train))
RMSE_test=sqrt(immse(gt_test,pred_test))

mdl = fitlm(pred_train(:),gt_train(:));
mdl2 = fitlm(pred_test(:),gt_test(:));

R2_distr=mdl.Rsquared.Ordinary;
R2_distr2=mdl2.Rsquared.Ordinary;



figure(11)
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


% %for 2NNs
% n=2;
% pred_2NN_all=all_data(:,2:(2+n-1));
% disagreement_2NN=zeros(all_data_size,n);
% %disagreement between NNs (root MSE value)
% for i=1:n
% disagreement_2NN(:,i)=sqrt(sum((pred_2NN_all(:,i)*ones(1,n)-pred_2NN_all).^2,2)/(n-1));
% end
% 
% disagreement_2NN_all=mean(disagreement_2NN,2);
% [~,index]=min(disagreement_2NN,[],2);
%  final_predictions_2NN=zeros(all_data_size,1);
% for j=1:all_data_size
%     final_predictions_2NN(j)=all_data(j,index(j));
% end
% %error between predictions and ground truths
% precision_2NN_all=abs(final_predictions_2NN-gt_all)+1e-50;

%for 3 NNs
n=3;
pred_3NN_all=all_data(:,2:(2+n-1));
disagreement_3NN=zeros(all_data_size,n);
%disagreement between NNs (root MSE value)
for i=1:n
disagreement_3NN(:,i)=sqrt(sum((pred_3NN_all(:,i)*ones(1,n)-pred_3NN_all).^2,2)/(n-1));
end

disagreement_3NN_all=mean(disagreement_3NN,2);
[~,index]=min(disagreement_3NN,[],2);
 final_predictions_3NN=zeros(all_data_size,1);
for j=1:all_data_size
    final_predictions_3NN(j)=all_data(j,index(j));
end
%error between predictions and ground truths
precision_3NN_all=abs(final_predictions_3NN-gt_all)+1e-50;

pred_train=final_predictions_3NN(1:train_size);
pred_test=final_predictions_3NN(train_size+1:end);
RMSE_train3=sqrt(immse(gt_train,pred_train))
RMSE_test3=sqrt(immse(gt_test,pred_test))

mdl = fitlm(pred_train(:),gt_train(:));
mdl2 = fitlm(pred_test(:),gt_test(:));
R2_distr=mdl.Rsquared.Ordinary
R2_distr2=mdl2.Rsquared.Ordinary

% %for 4 NNs
% n=4;
% pred_4NN_all=all_data(:,2:(2+n-1));
% disagreement_4NN=zeros(all_data_size,n);
% %disagreement between NNs (root MSE value)
% for i=1:n
% disagreement_4NN(:,i)=sqrt(sum((pred_4NN_all(:,i)*ones(1,n)-pred_4NN_all).^2,2)/(n-1));
% end
% 
% disagreement_4NN_all=mean(disagreement_4NN,2);
% [~,index]=min(disagreement_4NN,[],2);
%  final_predictions_4NN=zeros(all_data_size,1);
% for j=1:all_data_size
%     final_predictions_4NN(j)=all_data(j,index(j));
% end
% %error between predictions and ground truths
% precision_4NN_all=abs(final_predictions_4NN-gt_all)+1e-50;
% 
% %for 5 NNs
% n=5;
% pred_5NN_all=all_data(:,2:(2+n-1));
% disagreement_5NN=zeros(all_data_size,n);
% %disagreement between NNs (root MSE value)
% for i=1:n
% disagreement_5NN(:,i)=sqrt(sum((pred_5NN_all(:,i)*ones(1,n)-pred_5NN_all).^2,2)/(n-1));
% end
% 
% disagreement_5NN_all=mean(disagreement_5NN,2);
% [~,index]=min(disagreement_5NN,[],2);
%  final_predictions_5NN=zeros(all_data_size,1);
% for j=1:all_data_size
%     final_predictions_5NN(j)=all_data(j,index(j));
% end
% %error between predictions and ground truths
% precision_5NN_all=abs(final_predictions_5NN-gt_all)+1e-50;
% 
% %for 6 NNs
% n=6;
% pred_6NN_all=all_data(:,2:(2+n-1));
% disagreement_6NN=zeros(all_data_size,n);
% %disagreement between NNs (root MSE value)
% for i=1:n
% disagreement_6NN(:,i)=sqrt(sum((pred_6NN_all(:,i)*ones(1,n)-pred_6NN_all).^2,2)/(n-1));
% end
% 
% disagreement_6NN_all=mean(disagreement_6NN,2);
% [~,index]=min(disagreement_6NN,[],2);
%  final_predictions_6NN=zeros(all_data_size,1);
% for j=1:all_data_size
%     final_predictions_6NN(j)=all_data(j,index(j));
% end
% %error between predictions and ground truths
% precision_6NN_all=abs(final_predictions_6NN-gt_all)+1e-50;
% 
% %for 7 NNs
% n=7;
% pred_7NN_all=all_data(:,2:(2+n-1));
% disagreement_7NN=zeros(all_data_size,n);
% %disagreement between NNs (root MSE value)
% for i=1:n
% disagreement_7NN(:,i)=sqrt(sum((pred_7NN_all(:,i)*ones(1,n)-pred_7NN_all).^2,2)/(n-1)); %RMSE
% end
% 
% disagreement_7NN_all=mean(disagreement_7NN,2);
% [~,index]=min(disagreement_7NN,[],2);
%  final_predictions_7NN=zeros(all_data_size,1);
% for j=1:all_data_size
%     final_predictions_7NN(j)=pred_7NN_all(j,index(j));
% end
% %error between predictions and ground truths
% precision_7NN_all=abs(final_predictions_7NN-gt_all);



pred_train=final_predictions_3NN(1:train_size);
pred_test=final_predictions_3NN(train_size+1:end);
RMSE_train3=sqrt(immse(gt_train,pred_train))
RMSE_test3=sqrt(immse(gt_test,pred_test))

mdl = fitlm(pred_train(:),gt_train(:));
mdl2 = fitlm(pred_test(:),gt_test(:));

R2_distr=mdl.Rsquared.Ordinary;
R2_distr2=mdl2.Rsquared.Ordinary;





figure(4)

%plot error plot for all test data

 dis=disagreement_3NN_all(train_size+1:end);
 pres=precision_3NN_all(train_size+1:end);
 idx=(pres>1e-50);
 log_dis=log(dis(idx));
 log_pres=log(pres(idx));
 %yval=linspace(-18,0,10);
 %xval=-5.0*ones(1,10);
 %plot(xval,yval,'r','LineWidth',2)
hold on
scatter(log_dis,log_pres, 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[65,138,179]/255)
%scatter(log(disagreement_3NN_all(train_size+1:end)), log(precision_3NN_all(train_size+1:end)), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[65,138,179]/255)
 %scatter(log(disagreement_4NN_all), log(precision_4NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[166,183,39]/255)
 %scatter(log(disagreement_5NN_all), log(precision_5NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[246,146,0]/255)
%scatter(log(disagreement_6NN_all), log(precision_6NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[223,83,39]/255)
 %scatter(log(disagreement_7NN_all(train_size+1:end)), log(precision_7NN_all(train_size+1:end)), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[131,131,131]/255)
[x,y,err]=plotBinAve(log_dis,log_pres, 0.4);
%[x,y,err]=plotBinAve(log(disagreement_3NN_all(train_size+1:end)), log(precision_3NN_all(train_size+1:end)), 0.4);
y_plot = y;x_plot = x;err_plot=err;
y_plot(isnan(y)) = [];
x_plot(isnan(y)) = [];
err_plot(isnan(y))=[];
errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[65,138,179]/255, 'LineWidth',2)

% [x,y,err]=plotBinAve(log(disagreement_4NN_all(train_size+1:end)), log(precision_4NN_all(train_size+1:end)), 0.8);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[166,183,39]/255, 'LineWidth',2)
% 
% 
% [x,y,err]=plotBinAve(log(disagreement_5NN_all(train_size+1:end)), log(precision_5NN_all(train_size+1:end)), 0.8);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% %plot(x_plot,y_plot,'k-')
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[246,146,0]/255, 'LineWidth',2)
% 
% [x,y,err]=plotBinAve(log(disagreement_6NN_all(train_size+1:end)), log(precision_6NN_all(train_size+1:end)), 0.8);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% %plot(x_plot,y_plot,'k-')
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[223,83,39]/255, 'LineWidth',2)
% 
% 
% [x,y,err]=plotBinAve(log(disagreement_7NN_all(train_size+1:end)), log(precision_7NN_all(train_size+1:end)), 0.4);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[131,131,131]/255, 'LineWidth',2)

xlim([-10,-3.5])
ylim([-14,-2])
xlabel('Log(disagreement in predictions)','FontSize', 15) 
ylabel('Log(error of final prediction)','FontSize', 15)
title('3 NNs')
set(gca,'FontSize',15)
set(gca,'LineWidth',2)
%axis square
box on


% figure(5)
% 
% %plot error plot for all test data
% 
% % dis=[disagreement_div_test;disagreement_div_test_new];
% % pres=[precision_test;precision_test_new];
% % dis=disagreement_div_test;
% % pres=precision_test;
% % idx=(dis>1e-5);
% % log_dis=log(dis(idx));
% % log_pres=log(pres(idx));
% % yval=linspace(-14,0,10);
% % xval=-2.5*ones(1,10);
% hold on
% % scatter(log(disagreement_3NN_all), log(precision_3NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[65,138,179]/255)
% % scatter(log(disagreement_4NN_all), log(precision_4NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[166,183,39]/255)
% % scatter(log(disagreement_5NN_all), log(precision_5NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[246,146,0]/255)
% % scatter(log(disagreement_6NN_all), log(precision_6NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[223,83,39]/255)
% % scatter(log(disagreement_7NN_all), log(precision_7NN_all), 'o','filled','MarkerFaceAlpha',5/8,'MarkerFaceColor',[131,131,131]/255)
% 
% [x,y,err]=plotBinAve(log(disagreement_3NN_all(train_size+1:end)), log(precision_3NN_all(train_size+1:end)), 0.6);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% %plot(x_plot,y_plot,'k-')
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[65,138,179]/255, 'LineWidth',2)
% 
% [x,y,err]=plotBinAve(log(disagreement_4NN_all(train_size+1:end)), log(precision_4NN_all(train_size+1:end)), 0.6);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[166,183,39]/255, 'LineWidth',2)
% 
% 
% [x,y,err]=plotBinAve(log(disagreement_5NN_all(train_size+1:end)), log(precision_5NN_all(train_size+1:end)), 0.6);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% %plot(x_plot,y_plot,'k-')
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[246,146,0]/255, 'LineWidth',2)
% 
% [x,y,err]=plotBinAve(log(disagreement_6NN_all(train_size+1:end)), log(precision_6NN_all(train_size+1:end)), 0.6);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% %plot(x_plot,y_plot,'k-')
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[223,83,39]/255, 'LineWidth',2)
% 
% 
% 
% [x,y,err]=plotBinAve(log(disagreement_7NN_all(train_size+1:end)), log(precision_7NN_all(train_size+1:end)), 0.6);
% y_plot = y;x_plot = x;err_plot=err;
% y_plot(isnan(y)) = [];
% x_plot(isnan(y)) = [];
% err_plot(isnan(y))=[];
% errorbar(x_plot,y_plot,err_plot,'MarkerEdgeColor','none','MarkerFaceColor',[131,131,131]/255, 'LineWidth',2)
% 
% 
% %xlim([-10,-1.5])
% %ylim([-10,0])
% xlabel('Log(disagreement in predictions)','FontSize', 15) 
% ylabel('Log(error of final prediction)','FontSize', 15)
% set(gca,'FontSize',15)
% set(gca,'LineWidth',2)
% %axis square
% box on


