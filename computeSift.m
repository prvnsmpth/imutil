% The SIFT algorithm
% Given an image, returns SIFT keypoints
function keypoints = computeSift(I, options)

	% some flags
	showDogs = options.showDogs;

	% normalize pixel values in [0, 1]
	% and double the image to form first level of first octave
	I = I - min(I(:));
	I = I / max(I(:));
	I = doubleImage(I);	

	[M N] = size(I);	

	% initial sigma
	initSigma = options.initSigma;

	% convolve image to bring smoothing level to 1.6	
	I = gaussianConv(I, sqrt(initSigma ^ 2 - 1));	

	% some contraints
	minsize = 10;
	maxZoom = 100000;
	keypoints = [];

	% collect all DoGs and blurred images
	if (showDogs)
		blurPyr = cell(round(log2(min(M, N))) + 5);
		dogPyr = cell(round(log2(min(M, N))) + 5);
	end

	zoomLevel = 1;
	counter = 1;
	while (size(I, 1) > minsize && size(I, 2) > minsize && zoomLevel < maxZoom)
		
		% get keypoints for this octave
		[keys, I, I_blur, I_dog] = computeKeypoints(I, options, zoomLevel);
		keypoints = [keypoints; keys];		
		
		% resample image to half the size
		I = sampleImage(I);		

		if (showDogs)
			blurPyr{counter} = I_blur;	
			dogPyr{counter} = I_dog;
		end		
		
		fprintf('Octave complete %d, image size: (%d, %d)\n', zoomLevel, size(I, 1), size(I, 2));	
		zoomLevel = 2 * zoomLevel;
		counter = counter + 1;	

	end

	% show the blurs and the DoGs
	if (showDogs)
		plotImg(blurPyr(~cellfun('isempty', blurPyr)));
		plotImg(dogPyr(~cellfun('isempty', dogPyr)));
	end