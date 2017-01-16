function [ auroc ] = cnn_eval_histo(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.fileloc = '/home/veda/ColonHistology/CRCHistoPhenotypes_2016_04_28/Classification/';

opts.train.learningRate = [0.01*ones(1,60) 0.001*ones(1,40) 0.0001*ones(1,20)] ;
opts.train.weightDecay = 0.0005 ;
 
opts.expDir = fullfile('data', sprintf('histo')) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','histo') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.continue = false ;
opts.train.gpus = [] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

%load in imdb struct
opts.dataloc = '/home/veda/matconvnet-1.0-beta15/histo/data/histo/';
imdb = load(strcat(opts.dataloc,'imdb.mat'),'images');
imdb.meta = load(strcat(opts.dataloc,'imdb.mat'),'meta');


%recreate network, with softmax layer end instead of softmaxloss layer end
net = cnn_histo_init_eval(opts);

%load test set
test = find(imdb.images.set == 3);
test_im = imdb.images.data;
test_im = test_im(:,:,:,test);
test_lab = imdb.images.labels;
test_lab = test_lab(:,test);

%subtract mean of test images
dataMean = mean(test_im, 4);
for i=1:(size(test_im(1,1,1,:)))
    test_im(:,:,:,i) = test_im(:,:,:,i) - dataMean;
end


%evaluate results using simplenn
res = vl_simplenn(net,test_im);
res = squeeze(res(15).x);
auroc = auc_roc(res,test_lab);

end


function lab = binarize_labels(labels,k)
    for q=1:length(labels)
        if labels(q) == k
            lab(q) = 1;
        else
            lab(q) = 0;
        end
    end
end

%Generates the average ROC for the four classes
function auc = auc_roc(predictions,labels)
    preds = double(squeeze(predictions));
    X = 0;
    y = 0;
    t = 0;
    f = 0;
    e = 0;
    i = 0;
    o = 0;
    k = 0;
    
    if(sum(binarize_labels(labels,1) > 0) && sum(binarize_labels(labels,1)) < length(labels))
        [X,y,t,f] = perfcurve(binarize_labels(labels,1),preds(1,:),1);
        k = k + 1;
    end
    
    if(sum(binarize_labels(labels,2) > 0) && sum(binarize_labels(labels,2)) < length(labels))
        [X,y,t,e] = perfcurve(binarize_labels(labels,2),preds(2,:),1);
        k = k + 1;
    end
    
    if(sum(binarize_labels(labels,3) > 0) && sum(binarize_labels(labels,3)) < length(labels))
        [X,y,t,i] = perfcurve(binarize_labels(labels,3),preds(3,:),1);
        k = k + 1;
    end
    
    if(sum(binarize_labels(labels,4) > 0) && sum(binarize_labels(labels,4)) < length(labels))
        [X,y,t,o] = perfcurve(binarize_labels(labels,4),preds(4,:),1);
        k = k +1;
    end
    
    auc = ((f + e + i + o)/k)*100;
    
end




