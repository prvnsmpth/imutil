% Finds maxima and minima points in the DoG array, computes gradient
% magnitudes and orientations for every layer in the blurred image stack,
% and finally calls orientationHist to compute the keypoint descriptors 
%
% Given
% 	I_dog : DoG array of images
%	I_blur : Gaussian-blurred images
%	InitSigma : initial value of the scale parameter
%	intSize : Length of an interval
%	zoomLevel : down-sampling factor
% Returns
%	keypoints : keypoint descriptors
function keypoints = findExtrema(I_dog, I_blur, zoomLevel, options)

	% get size
	[M N numDogs] = size(I_dog);
	[M N numBlurs] = size(I_blur);

	initSigma = options.initSigma;
	intSize = options.intSize;
	numSpatialBins = options.numSpatialBins;
	numOrtBins = options.numOrtBins;
	
	sigma0 = initSigma * 2 ^ (1 / intSize);
	thresh = (0.04 / intSize) / 2;

	% gradient magnitudes and orientations
	[gradMag, gradOrt] = computeGrad(I_blur);
		
	%------------------ local maxima and minima ------------------%
	% idx = siftlocalmax(I_dog, 0.8 * thresh);
	% idx = [idx, siftlocalmax(-I_dog, 0.8 * thresh)];
	idx = getMaxima(I_dog);
	idx = [idx, getMaxima(-I_dog)];
	
	[i, j, s] = ind2sub(size(I_dog), idx);
	y = i - 1;
	x = j - 1;
	s = s - 2;
	candKeys = [x(:)'; y(:)'; s(:)'];

	% get rid of points that are within 'rad' distance of border
	rad = 3 * sigma0 * 2 .^ (candKeys(3, :) / intSize) * numSpatialBins / 2;
    sel = find(...
			candKeys(1, :) - rad >= 1          & ...
			candKeys(1, :) + rad <= N & ...
			candKeys(2, :) - rad >= 1          & ...
			candKeys(2, :) + rad <= M );
    candKeys = candKeys(:, sel);

    %----------------- accurate keypoint localization ----------------%    
    % candKeys = siftrefinemx(candKeys, I_dog, -1, thresh, 10); 
    candKeys = filterKeypoints(candKeys, I_dog);   

    %---------------- orientation histogram -----------------%     
	candKeys = siftormx(candKeys, I_blur, intSize, -1, sigma0);

	% multiply co-ordinates with suitable factor to get
	% co-ordinates in the original image
	x     = round(zoomLevel / 2 * candKeys(1, :));
	y     = round(zoomLevel / 2 * candKeys(2, :));
	sigma = zoomLevel / 2 * sigma0 * 2 .^ (candKeys(3, :) / intSize);
	ptDesc = [x(:)'; y(:)'; sigma(:)'; candKeys(4, :)];
	
	%------------------- build descriptor ---------------------%
	desc = siftdescriptor(I_blur, candKeys, sigma0, intSize, -1, ...
				'Magnif', 3, ...
				'NumSpatialBins', numSpatialBins, ...
				'NumOrientBins', numOrtBins);

	% This is the matrix of keypoint descriptors
	% Each row is a keypoint descriptor
	keypoints = [ptDesc; desc]';

