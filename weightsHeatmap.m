load weights.txt

[row, col]=size(weights)

for i=1:col
   [vmax, idx]=max(weights(:,i));
end



%HeatMap(weights)
colormap(autumn(5))
imagesc(weights)
colorbar
