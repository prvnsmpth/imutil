% Get candidate keypoints for an octave, by computing the blurred images
% and the DoGs and determining the scale-space extrema in one octave
%
% Given
%	I : The image using which the stack of blurred images is to be created
%	initSigma : Initial value of the scale parameter
%	zoomLevel : down-sampling factor
% Returns
%	keypoints : keypoint descriptors
% 	im : The blurred image to be re-sampled at next octave
%	I_blur : stack of blurred images (I_blur)
%	I_dog : stack of DoG images (I_dog)
function [keypoints, im, I_blur, I_dog] = computeKeypoints(I, options, zoomLevel)

	% size of image
	[M N] = size(I);

	% size of an interval
	intSize = options.intSize;
	initSigma = options.initSigma;

	% compute the sigma multiplier 'k'
	sigmaMult = 2 ^ (1 / intSize);

	% number of blur and DoG images in this octave
	numBlurs = intSize + 3;
	numDogs = intSize + 2;

	% initialize the image arrays
	I_blur = zeros(M, N, numBlurs);
	I_dog = zeros(M, N, numDogs);
	I_blur(:, :, 1) = I;

	% compute blurred images using incremental blurring
	prevSigma = initSigma;
	for i = 2:numBlurs
		sigmaDiff = prevSigma * sqrt(sigmaMult ^ 2 - 1);		
		I_blur(:, :, i) = gaussianConv(I_blur(:, :, i - 1), sigmaDiff);	
		prevSigma = prevSigma * sigmaMult;
	end

	% image to be resampled at next level is two images from top of stack
	im = I_blur(:, :, numBlurs - 2);

	% compute DoGs
	for i = 1:numDogs
		I_dog(:, :, i) = I_blur(:, :, i + 1) - I_blur(:, :, i);
	end

	% find extrema in compute DoGs
	options.numSpatialBins = 4;
	options.numOrtBins = 8;
	keypoints = findExtrema(I_dog, I_blur, zoomLevel, options);