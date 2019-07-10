% demonstration of style transfer with the Local Laplacian Filter
% This implementation does not include the color transfer
% 
% mathieu.aubry@m4x.org March 2014

%% load images 
I=imread('images/lounge.png');
M=imread('images/ruins.png');
I=rgb2gray(double(I)./255);
M=rgb2gray(double(M)./255);

%% main computation
tic
[O Og]=style_transfer(I,M,10,4);
toc

%% show results
figure;
imshow(I);title('Input photograph');
figure;
imshow(O);title('New style');
