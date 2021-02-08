function [x,y,stdev]=plotBinAve(ML,rsq,interv)
x = (min(ML(:))*1.05):interv:(max(ML(:))*0.95);
[n,bin] = histc(ML(:),x);

y = zeros(1,max(bin));
stdev= zeros(1,max(bin));
for i =  1:length(n)
    temp = rsq(bin==i);
    temp_noNaN = temp(~isnan(temp));
    y(i) = mean(temp(~isnan(temp_noNaN)));
    stdev(i) = std(temp(~isnan(temp_noNaN)));
end
x = x+interv./2;

end