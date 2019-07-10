clear all;

addpath('./smoothing_method/L0smoothing_SIG_11');
addpath('./smoothing_method/RollingGuidanceFilter_Matlab_eccv_14');
addpath('./smoothing_method/RTV_sig_12');
addpath('./smoothing_method/shockFilter');
addpath('./smoothing_method/WLS_tog_08');
addpath('./smoothing_method/WMF_cvpr_14');
addpath('./smoothing_method/detail_enhancement_WLS_tog_08');
addpath('./smoothing_method/fast_LLF_tog_14');
addpath('./smoothing_method/PencilDrawing_NPAR_12');

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

    % % fast_LLF enhancement 
    % for m = 2:8
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     patch_double = im2double(patch);
    %     patch_gray=rgb2gray(patch_double);
    %     I_ratio=patch_double./repmat(patch_gray+eps,[1 1 3]);

    %     fact = m;
    %     sigma=0.1;
    %     N=10;
    %     I_enhanced=llf(patch_gray,sigma,fact,N);
    %     patch_filtered=repmat(I_enhanced,[1 1 3]).*I_ratio;

    %     filename_label = sprintf('%s/%s-fast_LLFenhancement-%d.png',output_directory,secondList(n).name(1:end-10),m);        
    %     imwrite(patch_filtered,filename_label);
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

    % % fast_LLF enhancement with a general remapping function
    % for m = 1:1
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     patch_double = im2double(patch);
    %     patch_gray=rgb2gray(patch_double);
    %     I_ratio=patch_double./repmat(patch_gray+eps,[1 1 3]);

    %     N=20;
    %     I_enhanced=llf_general(patch_gray,@remapping_function,N);
    %     patch_filtered=repmat(I_enhanced,[1 1 3]).*I_ratio;

    %     filename_label = sprintf('%s/%s-fast_LLFenhancementgeneral.png',output_directory,secondList(n).name(1:end-10));
    %     imwrite(patch_filtered,filename_label);
    % end
    
    % % WLS enhancement
    % for m = 1:1
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));

    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     patch = im2double(patch);

    %     cform = makecform('srgb2lab');
    %     lab = applycform(patch, cform);
    %     L = lab(:,:,1);

    %     %% Filter
    %     L0 = wlsFilter(L, 0.125, 1.2);
    %     L1 = wlsFilter(L, 0.50,  1.2);

    %     %% Fine
    %     val0 = 25;
    %     val1 = 1;
    %     val2 = 1;
    %     exposure = 1.0;
    %     saturation = 1.1;
    %     gamma = 1.0;

    %     fine = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %     %% Medium
    %     val0 = 1;
    %     val1 = 40;
    %     val2 = 1;
    %     exposure = 1.0;
    %     saturation = 1.1;
    %     gamma = 1.0;

    %     med = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %     %% Coarse
    %     val0 = 4;
    %     val1 = 1;
    %     val2 = 15;
    %     exposure = 1.10;
    %     saturation = 1.1;
    %     gamma = 1.0;

    %     coarse = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %     patch_filtered=(coarse+med+fine)/3;

    %     % filename_label = sprintf('%s/%s_%d_%d.png',output_directory,secondList(n).name(1:end-4),height,width);
    %     filename_label = sprintf('%s/%s-WLSenhancement.png',output_directory,secondList(n).name(1:end-10));
    %     imwrite(patch_filtered,filename_label);
    % end

    % % color pencil drawing
    % for m = 1:1
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));
    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     patch_filtered = PencilDrawing(patch, 8, 1, 8, 1, 1);

    %     filename_label = sprintf('%s/%s-pencilColor.png',output_directory,secondList(n).name(1:end-10));
    %     imwrite(patch_filtered,filename_label);
    % end

    % % fast LLF style transfer
    % for m = 1:1
    %     filename_input = sprintf('%s/%s.png',input_directory,secondList(n).name(1:end-4));
    %     patch = imread(filename_input);
    %     [height,width,channel] = size(patch);

    %     patch_filtered=double(patch);
    %     patch=rgb2gray(im2double(patch));
    %     patch_filtered(:,:,1)=style_transfer(patch,style_image,20,3);
    %     patch_filtered(:,:,2)=patch_filtered(:,:,1);
    %     patch_filtered(:,:,3)=patch_filtered(:,:,1);

    %     filename_label = sprintf('%s/%s-style.png',output_directory,secondList(n).name(1:end-10));
    %     imwrite(patch_filtered,filename_label);
    % end

end