%This is just a toy example!
function y=remapping_function(x)
   % y=(x-0.1).*(x>0.1)+(x+0.1).*(x<-0.1); %smoothing
    y=3.*x.*(abs(x)<0.1)+(x+0.2).*(x>0.1)+(x-0.2).*(x<-0.1); %enhancement
end