% Computes the orientation histogram using points from around
% a given point, and determines keypoints to be created depending
% on the peak values in the histogram
%
% Given
%	grad : M x N matrix containing gradient magnitudes
%	ort : M x N matrix containing gradient orientations
%	kpScale : The scale parameter used to blur this layer (in the octave)
%	x : row number of point
%	y : col number of point
%	zoomLevel : down-sampling factor
% Returns
%	keypoints : keypoint descriptors generated at (x, y)
function keypoints = orientationHist(grad, ort, kpScale, x, y, zoomLevel)

	[M N] = size(grad);

	% 36 bins covering -PI to PI
	num_bins = 36;
	ortHist = zeros(num_bins, 1);
	keypoints = [];

	% std of gaussian weight-window, and the radius
	sig = 1.5 * kpScale;
	radius = 3 * sig;
	x = round(x);
	y = round(y);

	% window corners
	a = round(max(x - radius, 3));
	b = round(max(y - radius, 3));
	c = round(min(x + radius, M - 2));
	d = round(min(y + radius, N - 2));

	% iterate through all points in the window
	for i = a:c
		for j = b:d
			distnc =  norm([i j] - [x y]);
			gradVal = grad(i, j);
			if (distnc < radius && gradVal > 0)				
				ortAng = ort(i, j);
				binNo = round(num_bins * (ortAng + pi) / (2 * pi));
				if (binNo == 0)
					binNo = num_bins;
				end
				w = exp(-1 * distnc * distnc / ( 2 * sig * sig ));	
				ortHist(binNo) = ortHist(binNo) + w * gradVal;			
			end
		end
	end

	% smooth the histogram
	ortHist = smoothHist(ortHist);

	% the threshold for creating a keypoint is 80% of the 
	% highest peak in the histogram
	thresh = 0.8 * max(ortHist);

	% iterate through the histogram, looking for
	% peaks that are local maxima and higher than threshold
	for i = 1:num_bins
		if (i == 1)
			prev = num_bins;
		else
			prev = i - 1;
		end
		if (i == num_bins)
			next = 1;
		else
			next = i + 1;
		end
		if (ortHist(i) > ortHist(prev) && ortHist(i) > ortHist(next) ... 
			&& ortHist(i) > thresh)

			% make a parabolic fit over the three histogram bars
			peakVal = fitParabola(ortHist(prev), ortHist(i), ortHist(next));

			% compute orientation angle and create a keypoint at (x, y)
			ortAngle = ( ((i + peakVal) / num_bins) * 2 * pi ) - pi;

			% build the descriptor for this point
			desc = buildDesc(grad, ort, kpScale, x, y, ortAng, zoomLevel);
			keypoints = [keypoints; desc];

		end
	end

