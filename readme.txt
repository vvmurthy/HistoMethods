Major edit 1 : 16 Jan 2017
1. Provided 2 setups re: classification replication 
	a. one is in Theano / Lasagne 
	b. one is in matconvnet (original paper)
2. The matconvent version was developed around beta15, newer versions 
may be supported
3. Both methods follow the architecture outlined in:
K. Sirinukunwattana, S.E.A. Raza, Y.W Tsang, I.A. Cree, D.R.J. Snead, N.M. Rajpoot, ‘Locality 
Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer 
Histology Images,’ IEEE Transactions on Medical Imaging, 2016
4. The Lasagne version lacks the hsv augmentation performed between epochs
described in the paper. It also requires mean / std deviation centering
as required of Lasagne. 
5. Both methods (so far) meet the .78 F1 score / .91 AuROC outlined in the paper
for classification only
6. Both methods are not perfect in terms of replication and lack some of the
oversampling of nuclei described in the paper (still work in progress)
