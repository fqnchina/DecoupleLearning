clear all;

addpath('./smoothing_method/L0smoothing_SIG_11');
addpath('./smoothing_method/RollingGuidanceFilter_Matlab_eccv_14');
addpath('./smoothing_method/RTV_sig_12');
addpath('./smoothing_method/shockFilter');
addpath('./smoothing_method/WLS_tog_08');
addpath('./smoothing_method/WMF_cvpr_14');

input_directory = '/mnt/data/VOC2012_input/';
%%%%% change the output directory according to the image operator %%%%%
output_directory = '/mnt/data/VOC2012_L0smooth_output/';
%%%%% for the restoration tasks, the input images have to stored either, which is in the following directory %%%%%
% input_directory_save = '/mnt/data/VOC2012_L0smooth_input_save/';

secondList = dir([input_directory '/*-input.png']);
parpool(24);
parfor n = 1:length(secondList)
    % L0smooth
    for m = 1:7
        filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

        patch = imread(filename_input);
        [height,width,channel] = size(patch);

        R = [-log(10) log(10)];
        random = rand()*range(R)+min(R);
        lambda = exp(random) * 0.02;

        patch_filtered = L0Smoothing(patch, lambda);

        filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),lambda,height,width);
        imwrite(patch_filtered,filename_output);
    end

    % WLS
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);
    %     patch = im2double(patch);

    %     R = [-log(10) log(10)];
    %     random = rand()*range(R)+min(R);
    %     lambda = exp(random) * 1;

    %     patch_filtered = patch;
    %     patch_filtered(:,:,1) = wlsFilter(patch(:,:,1),lambda);
    %     patch_filtered(:,:,2) = wlsFilter(patch(:,:,2),lambda);
    %     patch_filtered(:,:,3) = wlsFilter(patch(:,:,3),lambda);

    %     filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),lambda,height,width);
    %     imwrite(patch_filtered,filename_output);
    % end

    % RTV
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     R = [-log(5) log(5)];
    %     random = rand()*range(R)+min(R);
    %     lambda = exp(random) * 0.01;

    %     patch_filtered = tsmooth(patch,lambda);

    %     filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),lambda,height,width);
    %     imwrite(patch_filtered,filename_output);
    % end

    % WMF
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);
    %     patch = im2double(patch);

    %     R = [1 10];
    %     r = rand()*range(R)+min(R);

    %     patch_filtered = jointWMF(patch,patch,r,25.5,256,256,1,'exp');

    %     filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),r,height,width);
    %     imwrite(patch_filtered,filename_output);
    % end

    % RGF
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     % if exist(filename_output, 'file')
    %     %     continue;
    %     % end
    %     % fileName = sprintf('%s/%s.jpg',input_directory,secondList(n).name(1:end-4))

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);
    %     patch = im2double(patch);

    %     R = [1 10];
    %     sigma = rand()*range(R)+min(R);

    %     patch_filtered = RollingGuidanceFilter(patch,sigma,0.05,4);

    %     filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),sigma,height,width);
    %     imwrite(patch_filtered,filename_output);
    % end

    % shock filter
    % for m = 1:1
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);
    %     patch = im2double(patch);
    %     patch_ycbcr = rgb2ycbcr(patch); 
    %     patch_y = patch_ycbcr(:,:,1);

    %     outputImg_y=shock_filter(patch_y,15,0.1,1,'org');

    %     patch_ycbcr(:,:,1) = outputImg_y;
    %     patch_filtered = ycbcr2rgb(patch_ycbcr);

    %     filename_output = sprintf('%s/%s_%0.5f_%d_%d.png',output_directory,secondList(n).name(1:end-4),1,height,width);
    %     imwrite(patch_filtered,filename_output);
    % end

    % % JPEG deblock
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = im2double(imread(filename_input));
    %     patch = rgb2ycbcr(patch); 
    %     patch = patch(:,:,1);
    %     [height,width,channel] = size(patch);

    %     R = [10 20];
    %     JPEG_Quality = rand()*range(R)+min(R);

    %     filename_input_save = sprintf('%s/%s_%0.5f_%d_%d.jpg',input_directory_save,secondList(n).name(1:end-4),JPEG_Quality,height,width);
    %     imwrite(patch,filename_input_save,'jpg','Quality',JPEG_Quality);
    %     filename_output = sprintf('%s/%s.png',output_directory,secondList(n).name(1:end-4));
    %     imwrite(patch,filename_output);
    % end

    % super resolution
    % for m = 2:4
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = im2double(imread(filename_input));
    %     patch = rgb2ycbcr(patch); 
    %     patch = patch(:,:,1);

    %     scale = 2;
    %     patch_crop = modcrop(patch,scale);
    %     [height,width] = size(patch_crop);

    %     patch_filtered = imresize(imresize(patch_crop,1/scale,'bicubic'),[height,width],'bicubic');

    %     filename_input_save = sprintf('%s/%s_%d_%d_%d.png',input_directory_save,secondList(n).name(1:end-4),scale,height,width);
    %     imwrite(patch_filtered,filename_input_save);
    %     filename_output = sprintf('%s/%s_%d.png',output_directory,secondList(n).name(1:end-4),scale);
    %     imwrite(patch_crop,filename_output);

    %     scale = 3;
    %     patch_crop = modcrop(patch,scale*2);
    %     [height,width] = size(patch_crop);

    %     patch_filtered = imresize(imresize(patch_crop,1/scale,'bicubic'),[height,width],'bicubic');

    %     filename_input_save = sprintf('%s/%s_%d_%d_%d.png',input_directory_save,secondList(n).name(1:end-4),scale,height,width);
    %     imwrite(patch_filtered,filename_input_save);
    %     filename_output = sprintf('%s/%s_%d.png',output_directory,secondList(n).name(1:end-4),scale);
    %     imwrite(patch_crop,filename_output);

    %     scale = 4;
    %     patch_crop = modcrop(patch,scale);
    %     [height,width] = size(patch_crop);

    %     patch_filtered = imresize(imresize(patch_crop,1/scale,'bicubic'),[height,width],'bicubic');

    %     filename_input_save = sprintf('%s/%s_%d_%d_%d.png',input_directory_save,secondList(n).name(1:end-4),scale,height,width);
    %     imwrite(patch_filtered,filename_input_save);
    %     filename_output = sprintf('%s/%s_%d.png',output_directory,secondList(n).name(1:end-4),scale);
    %     imwrite(patch_crop,filename_output);
    % end

    % % denoise
    % for m = 1:7
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = im2double(imread(filename_input));
    %     [height,width,channel] = size(patch);
    %     patch = rgb2gray(patch); 

    %     R = [10 70];
    %     sigma = rand()*range(R)+min(R);

    %     patch_filtered = patch + (sigma/255)*randn(size(patch));

    %     filename_input_save = sprintf('%s/%s_%0.5f_%d_%d.png',input_directory_save,secondList(n).name(1:end-4),sigma,height,width);
    %     imwrite(patch_filtered,filename_input_save);

    %     filename_output = sprintf('%s/%s.png',output_directory,secondList(n).name(1:end-4));
    %     imwrite(patch,filename_output);
    % end
end