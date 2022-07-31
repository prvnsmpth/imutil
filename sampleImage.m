% Reduce an image to 0.5x
function I_sampled = sampleImage(I)
	I_sampled = I(1:2:end, 1:2:end);