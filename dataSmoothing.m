close all
clear all
load('/home/lili/workspace/RNN/reverse2RNN_lili/data/freq16/seq5.txt')
timestamp=seq5(:,1);
joy_LR=seq5(:,2);
joy_BF=seq5(:,3);
linear_vel=seq5(:,4);
angular_vel=seq5(:,5);
N=size(linear_vel)
smooth_joy_LR=sgolayfilt(joy_LR,3,41);
smooth_joy_BF=sgolayfilt(joy_BF,3,41);
smooth_linear_vel=sgolayfilt(linear_vel,3,41);
smooth_angular_vel=sgolayfilt(angular_vel,3,41);

if 0
figure(1)

subplot(2,1,1)
plot(joy_LR)
title('seq2 joystick LR deflection before smooth')

subplot(2,1,2)
plot(smooth_joy_LR)
title('seq2 joystick LR deflection after smooth')

figure(2)

subplot(2,1,1)
plot(joy_BF)
title('seq1 joystick BF deflection before smooth')

subplot(2,1,2)
plot(smooth_joy_BF)
title('seq2 joystick BF deflection after smooth')


figure(3)

subplot(2,1,1)
plot(linear_vel)
title('seq2 linear velocity before smooth')

subplot(2,1,2)
plot(smooth_linear_vel)
title('seq2 linear velocity after smooth')

figure(4)

subplot(2,1,1)
plot(angular_vel)
title('seq2 angular velocity before smooth')

subplot(2,1,2)
plot(smooth_angular_vel)
title('seq2 angular velocity after smooth')

end

fileID2 = fopen('/home/lili/workspace/RNN/reverse2RNN_lili/data/freq16/smoothSeq4.txt', 'w');


for j=1:N
    fprintf(fileID2, '%s%s%s%s%s%s%s%s%s \n', num2str(timestamp(j)),',', num2str(smooth_joy_LR(j)),',', num2str(smooth_joy_BF(j)),',',num2str(smooth_linear_vel(j)), ',', num2str(smooth_angular_vel(j)));
end
fclose(fileID2);
