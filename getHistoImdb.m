function [ imdb ] = getHistoImdb(opts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fileloc = opts.fileloc;
r = 13;
k = 1;
data = zeros(27,27,3,100000);
data = single(data);
labels = zeros(1,100000);
labels = single(labels);

%Load in labels and images
for n=1:100
    str_n = num2str(n);
    full_im = imread(strcat(fileloc,'img',str_n,'/img',str_n,'.bmp'));
    full_im = single(full_im);
    
    
    epithelial = load(strcat(fileloc,'img',str_n,'/img',str_n,'_epithelial.mat'));
    e_lab = epithelial.detection;
    e_lab = int32(e_lab);
    
    inflammatory = load(strcat(fileloc,'img',str_n,'/img',str_n,'_inflammatory.mat'));
    i_lab = inflammatory.detection;
    i_lab = int32(i_lab);
    
    fibroblast = load(strcat(fileloc,'img',str_n,'/img',str_n,'_fibroblast.mat'));
    f_lab = fibroblast.detection;
    f_lab = int32(f_lab);
    
    others = load(strcat(fileloc,'img',str_n,'/img',str_n,'_others.mat'));
    o_lab = others.detection;
    o_lab = int32(o_lab);
    
    if(~isempty(f_lab))
        for i = 1:(length(f_lab(:,1)))
            y = f_lab(i,1);
            x = f_lab(i,2);
            
            if(x > 26) && (x < 473) && (y > 26) && (y < 473)
                im = full_im(x-r:x+r,y-r:y+r,:,:);
                data(:,:,:,k) = im;
                labels(k) = 1;
                k = k+1;
            end
        end
    end
    
    if(~isempty(e_lab))
        for i = 1:(length(e_lab(:,1)))
            y = e_lab(i,1);
            x = e_lab(i,2);
            if(x > 26) && (x < 473) && (y > 26) && (y < 473)
                im = full_im(x-r:x+r,y-r:y+r,:,:);
                data(:,:,:,k) = im;
                labels(k) = 2;
                k = k+1;
            end
        end
    end
    
    if(~isempty(i_lab))
        for i = 1:(length(i_lab(:,1)))
            y = i_lab(i,1);
            x = i_lab(i,2);
            if(x > 26) && (x < 473) && (y > 26) && (y < 473)
                im = full_im(x-r:x+r,y-r:y+r,:,:);
                data(:,:,:,k) = im;
                labels(k) = 3;
                k = k+1;
            end
        end
    end
    
    if(~isempty(o_lab))
        for i = 1:(length(o_lab(:,1)))
            y = o_lab(i,1);
            x = o_lab(i,2);
            if(x > 26) && (x < 473) && (y > 26) && (y < 473)
                im = full_im(x-r:x+r,y-r:y+r,:,:);
                data(:,:,:,k) = im;
                labels(k) = 4;
                k = k+1;
            end
        end
    end
       
    
end
k = k-1;    
data = data(:,:,:,1:k);
labels = labels(1:k);

%make splits for train, val and test
set = zeros(1,k);
set = uint8(set);
for i=1:k
    splitr = rand;
    if(i > 15000)
        set(i) = 2;
    elseif(i < 2000)
        set(i) = 3;
    else
        set(i) = 1;
    end
end


%subtract mean
dataMean = mean(data(:,:,:,set == 1), 4);
for i=1:k
    data(:,:,:,i) = data(:,:,:,i) - dataMean;
end

%set imdb
imdb.images.data = data ;
imdb.images.set = set;
imdb.images.labels = labels;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = {'fibroblast', 'epithelial', 'inflammatory','other'};


end

