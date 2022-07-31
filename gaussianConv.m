% Convolve a given image with a gaussian kernel
function I_filt = gaussianConv(I, sig)

	% kernel size
	sz = round([1 1] * (6 * sig + 1)); 

	% create filter ...
	F = fspecial('gaussian', sz, sig);

	% .. and do it
	I_filt = imfilter(I, F, 'same');