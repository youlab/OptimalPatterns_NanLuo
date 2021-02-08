clear all; close all; clc

file = xlsread('Training_sets_2.1.csv');
idx=file(:,end)>0;
sum(idx)
data=file(idx,:);
d_length=size(data,1);
training_length=floor(d_length*0.9);
test_length=d_length-training_length;

%Max_value=ones(d_length,1)*max(file,[],1);
%norm_data=file./Max_value; %biomass data normalization 
log_output=log(data(:,end));
normalization_factor=ceil(max(log_output));
norm_log_output=log_output/normalization_factor;
log_data1=[data(:,1:9), norm_log_output];
rng default
idx=randperm(training_length);
saved_data_training=log_data1(idx,:); %random permutation of the trainign data sets
saved_data_test=log_data1((training_length)+1:end,:);

figure(1)
subplot(1,2,1)
histogram(data(:,end))
xlabel('histogram of biomass','FontSize', 15) 
set(gca,'FontSize',15)
set(gca,'LineWidth',2)
subplot(1,2,2)
histogram(norm_log_output)
xlabel('histogram of normalized Log(biomass)','FontSize', 15) 
set(gca,'FontSize',15)
set(gca,'LineWidth',2)

WD_comb_matrix=file(1:351,8:9);
file_name = fullfile(pwd, sprintf('WD_comb_matrix.csv'));
dlmwrite(file_name,WD_comb_matrix);

file_name = fullfile(pwd, sprintf('training_data_normalized.csv'));
dlmwrite(file_name,saved_data_training);

file_name = fullfile(pwd, sprintf('test_data_normalized.csv'));
dlmwrite(file_name,saved_data_test);






