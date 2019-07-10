%   Multi-scale detail manipulation using WLS
%
%   This script computes a multi-scale edge-preserving decomposition
%   of an image, using the weighted least squares(WLS) optimization
%   framework, as described in Farbman, Fattal, Lischinski, and Szeliski,
%   "Edge-Preserving Decompositions for Multi-Scale Tone and Detail
%   Manipulation", ACM Transactions on Graphics, 27(3), August 2008.
%
%   The decomposition is then used to produce several different images
%   by boosting the contrast of one scale at a time.

%% Load
clear all;
filename = 'flower.png';
rgb = double(imread(filename))/255;
cform = makecform('srgb2lab');
lab = applycform(rgb, cform);
L = lab(:,:,1);

%% Filter
tic
L0 = wlsFilter(L, 0.125, 1.2);
L1 = wlsFilter(L, 0.50,  1.2);
toc

%% Fine
val0 = 25;
val1 = 1;
val2 = 1;
exposure = 1.0;
saturation = 1.1;
gamma = 1.0;

fine = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);
imwrite(fine, 'fine.png');
imtool(fine);

%% Medium
val0 = 1;
val1 = 40;
val2 = 1;
exposure = 1.0;
saturation = 1.1;
gamma = 1.0;

med = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);
imwrite(med, 'medium.png');
imtool(med);

%% Coarse
val0 = 4;
val1 = 1;
val2 = 15;
exposure = 1.10;
saturation = 1.1;
gamma = 1.0;

coarse = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);
imwrite(coarse, 'coarse.png');
imtool(coarse);

