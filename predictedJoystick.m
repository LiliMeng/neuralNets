load('pedictedJoystickLR.txt')
load('pedictedJoystickBF.txt')
joyLR=pedictedJoystickLR(:,1);
joyBF=pedictedJoystickBF(:,1);

N=size(joyLR(:,1),1);

fileID3 = fopen('/home/lili/workspace/RNN/data/freq16/predictedJoystick.txt', 'w');

for i=1:N
    fprintf(fileID3, '%s %s %s\n', num2str(joyLR(i)),',',num2str(joyBF(i)));
end

fclose(fileID3);
