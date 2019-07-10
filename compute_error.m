%%%%%% uncomment for the jointly trained 10 operator model with real-time parameter tuning module %%%%%%
% types = {'L0smooth','WLS','RTV','RGF','WMF','fast_LLFenhancement','fast_LLFenhancementgeneral','WLSenhancement','style','pencilColor'};
% totalPara = {{'000200', '000431', '002000', '009283', '020000'},
% {'010000', '021544', '100000', '464159', '1000000'},
% {'000200', '000447', '001000', '002236', '005000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'2','3','5','7','8'},
% {'1.00000'},
% {'1.00000'},
% {'1.00000'},
% {'1.00000'}};



%%%%%% uncomment for the jointly trained 10 operator model %%%%%%
% types = {'L0smooth','WLS','RTV','RGF','WMF','shock','deblock-y','rain','SR-y','noise-gray'};
% totalPara = {{'000200', '000431', '002000', '009283', '020000'},
% {'010000', '021544', '100000', '464159', '1000000'},
% {'000200', '000447', '001000', '002236', '005000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'1.00000'},
% {'10','20'},
% {'10'},
% {'2','3','4'},
% {'15','25','50'}
% };



%%%%%% uncomment for the jointly trained 6 filtering based operator model %%%%%%
% types = {'L0smooth','WLS','RTV','RGF','WMF','shock'};
% totalPara = {{'000200', '000431', '002000', '009283', '020000'},
% {'010000', '021544', '100000', '464159', '1000000'},
% {'000200', '000447', '001000', '002236', '005000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'1.00000','3.25000','5.50000','7.75000','10.00000'},
% {'1.00000'}};



%%%%%% uncomment for the jointly trained 4 restoration operator model %%%%%%
% types = {'deblock-y','rain','SR-y','noise-gray'};
% totalPara = {{'10','20'},
% {'10'},
% {'2','3','4'},
% {'15','25','50'}
% };



%%%%%% uncomment for the individually trained single operator model %%%%%%
% types = {'L0smooth'};
% totalPara = {{'000200', '000431', '002000', '009283', '020000'}};
% types = {'WLS'};
% totalPara = {{'010000', '021544', '100000', '464159', '1000000'}};
% types = {'RTV'};
% totalPara = {{'000200', '000447', '001000', '002236', '005000'}};
% types = {'RGF'};
% totalPara = {{'1.00000','3.25000','5.50000','7.75000','10.00000'}};
% types = {'WMF'};
% totalPara = {{'1.00000','3.25000','5.50000','7.75000','10.00000'}};
% types = {'shock'};
% totalPara = {{'1.00000'}};
% types = {'deblock-y'};
% totalPara = {{'10','20'}};
% types = {'rain'};
% totalPara = {{'10'}};
% types = {'SR-y'};
% totalPara = {{'2','3','4'}};
% types = {'noise-gray'};
% totalPara = {{'15','25','50'}};

inputDir = './results/';
for m = 1:length(totalPara)
    para = totalPara{m};
    disp(types{m});
    totalPSNR_all = 0;
    totalSSIM_all = 0;
    totalMSE_all = 0;
    count_all = 0;
    for k = 1:length(para)
        totalPSNR = 0;     
        totalSSIM = 0;
        totalMSE = 0;
        count = 0;

        disp(para{k});
        if length(para) ~=1
            searchDir = sprintf('%s*-%s-%s-predict.png',inputDir,types{m},para{k});
        else
            searchDir = sprintf('%s*-%s-predict.png',inputDir,types{m});
        end

        if strcmp(types{m}, 'rain')
            searchDir = sprintf('%s*_GT.png',inputDir);   
        end
        if strcmp(types{m}, 'shock')
            searchDir = sprintf('%s*-%s-%s-predict.png',inputDir,types{m},para{k});
        end

        s = dir(searchDir);
        for n = 1:length(s)
            inputName = [inputDir s(n,1).name];
            targetName = strrep(inputName,'-predict','');
            if strcmp(types{m}, 'rain')
                targetName = strrep(inputName,'_GT','_predict');
            end
            if strcmp(types{m}, 'SR-y')
                targetName = strrep(inputName,sprintf('SR-y-%s-predict',para{k}),sprintf('input-y-%s',para{k}));
            end
            if strcmp(types{m}, 'noise-gray')
                targetName = strrep(inputName,sprintf('noise-gray-%s-predict',para{k}),'label-gray');
            end
            if strcmp(types{m}, 'deblock-y')
                targetName = strrep(inputName,sprintf('deblock-y-%s-predict.png',para{k}),'label-y.png');
            end

            input = imread(inputName);
            target = imread(targetName);
            [height,width,channel] = size(input);

            [peaksnr, snr] = psnr(input, target);
            totalPSNR = totalPSNR + peaksnr;

            input = rgb2gray(input);
            target = rgb2gray(target);
            [ssimval, ssimmap] = ssim(input, target);
            totalSSIM = totalSSIM + ssimval;
            count = count + 1;
        end
        totalPSNR = totalPSNR/count;
        totalSSIM = totalSSIM/count;
        disp(sprintf('psnr: %f, ssim: %f',totalPSNR,totalSSIM));
        totalPSNR_all = totalPSNR_all + totalPSNR;
        totalSSIM_all = totalSSIM_all + totalSSIM;
        count_all = count_all + 1;
    end
    disp(sprintf('AVE psnr: %f, ssim: %f',totalPSNR_all/count_all,totalSSIM_all/count_all));
end