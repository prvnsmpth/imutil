function keypoints = orientHist(keys, initSigma)

	numBins = 36;
	ortHist = zeros(numBins, 1);

	[M N] = size(keys);

	for i = 1:N
		col = keys(1, i);
		row = keys(2, i);
		lay = keys(3, i);
		kpScale = 2;
	end

