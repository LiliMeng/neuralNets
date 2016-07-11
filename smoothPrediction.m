clear all
close all

%using SG Filter to smooth the data 

seq1=load('/home/lili/workspace/RNN/data/freq16/seq1.txt');
N1 =size(seq1);
N=N1(1);
timestamp=seq1(:,1);
LR=seq1(:,2);
BF=seq1(:,3);

linear_vel=seq1(:,4);
angular_vel=seq1(:,5);

sm_LR=sgolayfilt(LR,3,41);
sm_BF=sgolayfilt(BF,3,41);

sm_linear_vel=sgolayfilt(linear_vel,3,41);
sm_angular_vel=sgolayfilt(angular_vel,3,41);

figure(1)
subplot(2,2,1);
plot(LR);
title('Joy LR before smooth');
grid;

subplot(2,2,3);
plot(sm_LR);
title('Joy LR after smooth');
grid;

subplot(2,2,2);
plot(BF);
title('Joy BF before smooth');
grid;

subplot(2,2,4);
plot(sm_BF);
title('Joy BF after smooth');
grid;

figure(2)
subplot(2,2,1);
plot(linear_vel);
title('linear velocity before smooth');
grid;

subplot(2,2,3);
plot(sm_linear_vel);
title('linear velocity after smooth');
grid;

subplot(2,2,2);
plot(angular_vel);
title('angular velocity before smooth');
grid;

subplot(2,2,4);
plot(sm_angular_vel);
title('angular velocity after smooth');
grid;

fileID=fopen('/home/lili/workspace/RNN/data/freq16/seq1_smooth.txt','w');

for i =1:N
    fprintf(fileID, '%s%s%s%s%s%s%s%s%s\n', num2str(timestamp(i)),',',num2str(sm_LR(i)),',',num2str(sm_BF(i)), ',', num2str(sm_linear_vel(i)),',',num2str(sm_angular_vel(i)));
end

fclose(fileID);
