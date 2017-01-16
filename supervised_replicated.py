"""Analyzes the (pre-extracted) Nuclei of 
K. Sirinukunwattana, S.E.A. Raza, Y.W Tsang, I.A. Cree, D.R.J. Snead, N.M. Rajpoot, 
Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images
IEEE Transactions on Medical Imaging, 2016 (in press)

Using Lasagne and Theano setup

Differences from original version 
1. Images are mean / std deviation centered
2. Images are not modified in hsv colorspace between epochs


"""

from scipy import misc
import numpy as np 
import scipy.io as sio
import lasagne
import theano
import theano.tensor as T
import sklearn.metrics as m

def onehot_to_vector(labels):
    x = len(labels)
    vector = np.zeros((x,4))
    for k in range(x):
        vector[k,labels[k]] = 1
    return vector

def preds_to_binary(preds):
    binary = np.zeros((preds.shape[0],4))
    for n in range(0,preds.shape[0]):
        index = np.argmax(preds[n,:])
        binary[n,index] =1
    return binary

def roc_avg(train_order,train_preds):
    f = m.roc_auc_score(train_order[:,0],train_preds[:,0])
    e = m.roc_auc_score(train_order[:,1],train_preds[:,1])
    i = m.roc_auc_score(train_order[:,2],train_preds[:,2])
    o = m.roc_auc_score(train_order[:,3],train_preds[:,3])
    return (f + e + i + o) /4
    
def data_aug(images):
    for n in range(0,images.shape[0]):
        splitr = np.random.random()
        if(splitr < 0.05):
            images[n,0,:,:] = np.fliplr(images[n,0,:,:])
            images[n,1,:,:] = np.fliplr(images[n,1,:,:])
            images[n,2,:,:] = np.fliplr(images[n,2,:,:])
        elif(splitr > 0.05 and splitr < 0.1):
            images[n,0,:,:] = np.flipud(images[n,0,:,:])
            images[n,1,:,:] = np.flipud(images[n,1,:,:])
            images[n,2,:,:] = np.flipud(images[n,2,:,:])
        elif(splitr > 0.1 and splitr < 0.15):
            images[n,0,:,:] = np.rot90(images[n,0,:,:])
            images[n,1,:,:] = np.rot90(images[n,1,:,:])
            images[n,2,:,:] = np.rot90(images[n,2,:,:])
        elif(splitr > 0.15 and splitr < 0.2):
            images[n,0,:,:] = np.flipud(images[n,0,:,:])
            images[n,1,:,:] = np.flipud(images[n,1,:,:])
            images[n,2,:,:] = np.flipud(images[n,2,:,:])
            images[n,0,:,:] = np.rot90(images[n,0,:,:])
            images[n,1,:,:] = np.rot90(images[n,1,:,:])
            images[n,2,:,:] = np.rot90(images[n,2,:,:])
    return images            
    
def read_data():
    images = np.zeros((100,500,500,3), dtype=np.uint8)
    path = "/home/veda/histo_im/Classification/"
    for i in range(1,101):
        filename = "img" + str(i)
        fullpath = path + filename + "/" + filename + ".bmp" 
        images[i-1,:,:,:] = misc.imread(fullpath)
    
    return images

"Labels are in the form [epithelial, fibroblast, inflammatory, other]"
def create_dataset(images):
    data = np.zeros((25000,3,27,27)).astype(np.uint8)
    labels = np.zeros((25000,1)).astype(np.int8)
    nuclei = 0;
    r = 13
    no_fibroblast = 0
    no_epithelial = 0
    no_inflammatory = 0
    no_others = 0
    
    #take 1 500x500 image
    for k in range(0,100):
        
        if(k % 10 == 0):
            print("Loading Image " + str(k) + " of 100")
        
        
        image = images[k]
     
        fibroblast = sio.loadmat("/home/veda/histo_im/Classification/img" + str(k+1) + "/img" + 
                                 str(k+1) +"_fibroblast")
        coors = fibroblast["detection"]
        if(coors.shape[0] > 0):
            for i in range(0,coors.shape[0]):
                
                y = coors[i,0] -1
                y = np.round(y)
                y = y.astype(np.int32)
                x = coors[i,1] -1
                x = np.round(x)
                x = x.astype(np.int32)
                
                
                if(y > 25 and y < 472 and x > 25 and x < 472):
                    add = image[x-r:x+r+1,y-r:y+r+1,:]
                    add = np.swapaxes(np.swapaxes(add, 1, 2), 0, 1)
                    data[nuclei,:,:,:] = add;
                    labels[nuclei] = 0
                    nuclei=nuclei+1
                    no_fibroblast += 1
                
            
        epithelial = sio.loadmat("/home/veda/histo_im/Classification/img" + str(k+1) + "/img" + 
                                 str(k+1) +"_epithelial")
        coors = epithelial["detection"]
        if(coors.shape[0] > 0):
            for i in range(0,coors.shape[0]):
                
                y = coors[i,0] -1
                y = np.round(y)
                y = y.astype(np.int32)
                x = coors[i,1] -1
                x = np.round(x)
                x = x.astype(np.int32)
                
                
                if(y > 25 and y < 472 and x > 25 and x < 472):
                   add = image[x-r:x+r+1,y-r:y+r+1,:]
                   add = np.swapaxes(np.swapaxes(add, 1, 2), 0, 1)
                   data[nuclei,:,:,:] = add;
                   labels[nuclei] = 1
                   nuclei=nuclei+1
                   no_epithelial += 1
                    
        inflammatory = sio.loadmat("/home/veda/histo_im/Classification/img" + str(k+1) + "/img" + 
                                 str(k+1) +"_inflammatory")
        coors = inflammatory["detection"]
        if(coors.shape[0] > 0):
            for i in range(0,coors.shape[0]):
                
                y = coors[i,0] -1
                y = np.round(y)
                y = y.astype(np.int32)
                x = coors[i,1] -1
                x = np.round(x)
                x = x.astype(np.int32)
                
                
                if(y > 25 and y < 472 and x > 25 and x < 472):
                   add = image[x-r:x+r+1,y-r:y+r+1,:]
                   add = np.swapaxes(np.swapaxes(add, 1, 2), 0, 1)
                   data[nuclei,:,:,:] = add;
                   labels[nuclei] = 2
                   nuclei=nuclei+1
                   no_inflammatory += 1
          
        others = sio.loadmat("/home/veda/histo_im/Classification/img" + str(k+1) + "/img" + 
                                 str(k+1) +"_others")
        coors = others["detection"]
        if(coors.shape[0] > 0):
            for i in range(0,coors.shape[0]):
                
                y = coors[i,0] -1
                y = np.round(y)
                y = y.astype(np.int32)
                x = coors[i,1] -1
                x = np.round(x)
                x = x.astype(np.int32)
                
                
                if(y > 25 and y < 472 and x > 25 and x < 472):
                   add = image[x-r:x+r+1,y-r:y+r+1,:]
                   add = np.swapaxes(np.swapaxes(add, 1, 2), 0, 1)
                   data[nuclei,:,:,:] = add;
                   labels[nuclei] = 3
                   nuclei=nuclei+1
                   no_others += 1
    
                    
    print("Number of Fibroblasts: " + str(no_fibroblast)
    + " Number of Epithelial: " + str(no_epithelial) + 
    " Number of inflammatory: " + str(no_inflammatory) + 
    " Number of others: " + str(no_others))
    data = data[0:nuclei,:,:,:]
    labels = labels[0:nuclei]
    data = data.astype(np.float32)  

    
          
    labels = labels.astype(np.int32)
    return data, labels  
    
def data_split(data,labels):
    train_data = np.zeros((20000,3,27,27)).astype(np.float32)
    train_labels = np.zeros((20000,1)).astype(np.int8)
    val_data = np.zeros((5000,3,27,27)).astype(np.float32)
    val_labels = np.zeros((5000,1)).astype(np.int8)
    test_data = np.zeros((5000,3,27,27)).astype(np.float32)
    test_labels = np.zeros((5000,1)).astype(np.int8)
    testnum = 0
    valnum = 0
    trainnum = 0
    for n in range(0,labels.shape[0]):
        
        if(n % 5000 == 0):
            print("Splitting Image " + str(n) + " of " + str(labels.shape[0]))
        
        im = data[n,:,:,:]
        lab = labels[n]
        splitr = np.random.random();
        
        if(splitr < 0.2):
            test_data[testnum,:,:,:] = im
            test_labels[testnum] = lab
            testnum += 1
        elif(splitr > 0.85):
            val_data[valnum,:,:,:] = im
            val_labels[valnum] = lab
            valnum += 1
        else:
            train_data[trainnum,:,:,:] = im
            train_labels[trainnum] = lab
            trainnum += 1
    
    test_data = test_data[0:testnum,:,:,:]
    test_labels = np.squeeze(test_labels[0:testnum])
    val_data = val_data[0:valnum,:,:,:]
    val_labels = np.squeeze(val_labels[0:valnum])
    train_data = train_data[0:trainnum,:,:,:]
    train_labels = np.squeeze(train_labels[0:trainnum])
    
    #Subtract mean image
    mean_im = np.mean(train_data,axis=0)
    for k in range(0,train_data.shape[0]):
        train_data[k,:,:,:] = train_data[k,:,:,:] - mean_im
    for k in range(0,val_data.shape[0]):
        val_data[k,:,:,:] = val_data[k,:,:,:] - mean_im
    for k in range(0,test_data.shape[0]):
        test_data[k,:,:,:] = test_data[k,:,:,:] - mean_im
    
    mean = np.mean(train_data);
    std = np.std(train_data);
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def build_cnn():
    input_var = T.tensor4('inputs')
    network = lasagne.layers.InputLayer(shape=(None, 3, 27, 27),
    input_var=input_var)
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
    network, num_filters=36, filter_size=(4, 4),
    W = lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print(lasagne.layers.get_output_shape(network))
    
    network = lasagne.layers.Conv2DLayer(
    network, num_filters=48, filter_size=(3, 3),
    W = lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(network,
    num_units=512,W = lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))
    
    network = lasagne.layers.DropoutLayer(network,p=0.2)

    fc6 = lasagne.layers.DenseLayer(network,
    num_units=512,W = lasagne.init.Normal(),
    nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(fc6))
    
    fc6 = lasagne.layers.DropoutLayer(fc6,p=0.2)
    
    out = lasagne.layers.DenseLayer(fc6,
    num_units=4,
    nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(out))
    

    target_var = T.ivector('targets')
    activations = lasagne.layers.get_output(fc6)
    prediction = lasagne.layers.get_output(out)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    all_layers = lasagne.layers.get_all_layers(out)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0005
    loss = loss + l2_penalty
    lr = T.fscalar('lrate')
    

    params = lasagne.layers.get_all_params(out, trainable=True)
    updates = lasagne.updates.momentum(
    loss, params, learning_rate=lr, momentum=0.9)

    test_prediction = lasagne.layers.get_output(out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var, lr], [loss,prediction,activations], updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])
    
    
    return out, train_fn, val_fn

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
    if(batchsize > len(targets)):
        yield inputs, targets
        
        
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt] 
        
def main():    
    images = read_data()
    num_epochs =  120
    data, labels = create_dataset(images)
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(data,labels)
    out, train_fn, val_fn = build_cnn()
   
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    lrate = 0.01
    for epoch in range(num_epochs):
        
        if(epoch == 60):
            lrate = 0.001
        if(epoch == 120):
            lrate = 0.0001
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        rd_err = 0
        train_batches = 0
        train_preds = np.zeros((0,4))
        train_order = np.zeros((0,4))
        val_preds = np.zeros((0,4))
        val_order = np.zeros((0,4))
        test_preds = np.zeros((0,4))
        test_order = np.zeros((0,4))
        activations = np.zeros((0,512))
     
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            inputs = data_aug(inputs)
            rd_err, preds, act = train_fn(inputs, targets, lrate)
            activations = np.concatenate((activations,act))
            train_preds = np.concatenate((train_preds,preds))
            train_order = np.concatenate((train_order,onehot_to_vector(targets)))
            train_batches += 1
            train_err += rd_err
        
        print("Training ROC: " + str(roc_avg(train_order,train_preds)))
        print("Training F1: " + str(m.f1_score(train_order,preds_to_binary(train_preds),average='weighted'))) 
       
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, preds = val_fn(inputs, targets)
            val_preds = np.concatenate((val_preds,preds))
            val_order = np.concatenate((val_order,onehot_to_vector(targets)))
            val_err += err
            val_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} Completed".format(
        epoch + 1, num_epochs))
        print(" training loss:\t\t{:.6f}".format(train_err / train_batches))
        print(" validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("Validation ROC: " + str(roc_avg(val_order,val_preds))) 
        print("Validation F1: " + str(m.f1_score(val_order,preds_to_binary(val_preds),average='weighted'))) 
   
   # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, preds = val_fn(inputs, targets)
        test_preds = np.concatenate((test_preds,preds))
        test_order = np.concatenate((test_order,onehot_to_vector(targets)))
        test_err += err
        test_batches += 1
    print("Final results:")
    print(" test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("Test ROC: " + str(roc_avg(test_order,test_preds))) 
    print("Test F1: " + str(m.f1_score(test_order,preds_to_binary(test_preds)))) 
main()

