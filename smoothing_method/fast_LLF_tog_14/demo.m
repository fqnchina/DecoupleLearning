% demonstration of the Local Laplacian Filter
% 
% mathieu.aubry@m4x.org March 2014

%% import image
clear all;
name='demo-15';
I_rgb = imread(sprintf('images/%s.png',name));
I = rgb2gray(im2double(I_rgb));
I_ratio=double(I_rgb)./repmat(I,[1 1 3])./255;
    
    
%% image smoothing
sigma=0.2;
N=5;
fact=5 ;
tic
I_smoothed=llf(I,sigma,fact,N);
toc
I_smoothed=repmat(I_smoothed,[1 1 3]).*I_ratio;

imwrite(I_smoothed,'demo-15-fastLLF.png');

% %% image enhancement
% sigma=0.1;
% N=10;
% fact=5;
% tic
% I_enhanced=llf(I,sigma,fact,N);
% toc
% I_enhanced=repmat(I_enhanced,[1 1 3]).*I_ratio;
% 
% %% image enhancement using a general remapping function
% N=20;
% tic
% I_enhanced2=llf_general(I,@remapping_function,N);
% toc
% I_enhanced2=repmat(I_enhanced2,[1 1 3]).*I_ratio;
% 
% 
% % plot
% figure;
% subplot(2,2,1); imshow(I_rgb); title('Input photograph');
% subplot(2,2,2); imshow(I_enhanced); title('Edge-aware enhacement with LLF');
% subplot(2,2,3); imshow(I_smoothed); title('Edge-aware smoothing with LLF');
% subplot(2,2,4); imshow(I_enhanced2); title('Edge-aware enhancement with a general LLF');
   
