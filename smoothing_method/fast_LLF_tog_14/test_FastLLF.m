
I_rgb = imread('/mnt/data/MIT-input-fullsize/paper2-1.png');
I_rgb = imresize(I_rgb, [1280 720]);
I = rgb2gray(im2double(I_rgb));
I_ratio=double(I_rgb)./repmat(I,[1 1 3])./255;

for m = 1:100 
	tic;
	%% image smoothing
	sigma=0.2;
	N=5;
	fact=-1 ;

	I_smoothed=llf(I,sigma,fact,N);
	I_smoothed=repmat(I_smoothed,[1 1 3]).*I_ratio;
	toc;
end

figure,imshow(I_rgb);
figure,imshow(I_smoothed);

% QVGA 0.059066
% VGA 0.128067
% 720p 0.212031