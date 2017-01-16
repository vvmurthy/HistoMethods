function [net, info] = cnn_histo(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.fileloc = '/home/veda/histo_im/Classification/';

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

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

net = cnn_histo_init(opts) ;


%if exist(opts.imdbPath, 'file')
%  imdb = load(opts.imdbPath) ;
%else
  imdb = getHistoImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
%end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_histo(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 2)) ;
end
%--------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
splitr = rand;
if(splitr > 0.8)
    im = fliplr(im);
elseif(splitr < 0.8 && splitr >=0.6)
     im = flipud(im);
elseif(splitr < 0.6 && splitr >=0.55)
    im = rot90(im);
elseif(splitr < 0.55 && splitr >=0.5)
    im = hsv_perturb(im);
end
end

function im = hsv_perturb(im)
    for i=1:(length(im(1,1,1,:)))
        im(:,:,:,i) = per_im_perturb(im(:,:,:,i));
    end
end

%RGB2hsv only accepts 3D matrix, not 4D
%hence we need to wrap the function per batch
% so it runs x times, where x is batch size
function im = per_im_perturb(im)
    hsv = rgb2hsv(im);
    coeff1 = 0.95+rand*(1.05-0.95);
    coeff2 = 0.9+rand*(1.1-0.9);
    coeff3 = 0.9+rand*(1.1-0.9);
    hsv(:,:,1,:) = coeff1*hsv(:,:,1,:);
    hsv(:,:,2,:) = coeff2*hsv(:,:,2,:);
    hsv(:,:,3,:) = coeff3*hsv(:,:,3,:);
    im = hsv2rgb(hsv);
end


