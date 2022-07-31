function Hsmooth = smoothHist(H)

	Hsmooth = H;
	numBins = size(Hsmooth, 1);

	% run the smoothing 7 times
	for n = 1:7		
		Hsmooth(1) = ( Hsmooth(numBins) + Hsmooth(1) + Hsmooth(2) ) / 3;
		for i = 2:numBins-1
			Hsmooth(i) = ( Hsmooth(i - 1) + Hsmooth(i) + Hsmooth(i + 1) ) / 3;
		end
		Hsmooth(numBins) = ( Hsmooth(numBins - 1) + Hsmooth(numBins) + Hsmooth(1) ) / 3;
	end
