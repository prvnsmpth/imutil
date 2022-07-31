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

	% This is the matrix of keypoint descriptors
	% Each row is a keypoint descriptor
	keypoints = [];

	initSigma = options.initSigma;
	intSize = options.intSize;

	% binary matrix - 1 if extrema, 0 otherwise
	bitmap = zeros(M, N);

	% gradient magnitudes and orientations
	gradMag = zeros(M, N);
	gradOrt = zeros(M, N);
	count = 0;

	% iterate through the scales, and find points that are
	% either local maxima or local minima in the 26-point
	% neighbourhood
	for i = 2:numDogs-1

		% get the ith layer in the DoGs
		dog = I_dog(:, :, i);

		% compute gradient magnitudes and orientations for this layer
		[gradMag gradOrt] = computeGrad(I_blur(:, :, i));

		% loop over points not very close to the edge
		for j = 3:M-2
			for k = 3:N-2
				val = dog(j, k);
				rowSlice = [max(j - 1, 1):min(j + 1, M)];
				colSlice = [max(k - 1, 1):min(k + 1, N)];
				sliceAbove = I_dog(rowSlice, colSlice, i - 1);
				sliceAdj = I_dog(rowSlice, colSlice, i);
				sliceBelow = I_dog(rowSlice, colSlice, i + 1);
				nb = [sliceAbove(:); sliceAdj(:); sliceBelow(:)];
				if ( (isMaxima(val, nb) || isMinima(val, nb)) && checkEdgeResponse(dog, j, k) )
					count = count + 1;
					% obtain accurate keypoint locale (3D quadratic fit)
					[keyloc keyval] = fitQuadratic(I_dog, i, j, k);

					% check if extrema lies elsewhere, if yes, move to that point
					% and re-fit
					maxMoves = 5; newrow = j; newcol = k;
					while (1)
						if (maxMoves == 0)
							break;
						end
						maxMoves = maxMoves - 1;
						
						r = newrow; c = newcol;
						if ( keyloc(2) > 0.5 && newrow < M - 2 )
							newrow = newrow + 1;
						elseif ( keyloc(2) < -0.5 && newrow > 3 )
							newrow = newrow - 1;
						end

						if ( keyloc(3) > 0.5 && newcol < N - 2 )
							newcol = newcol + 1;
						elseif ( keyloc(3) < -0.5 && newcol > 3 )
							newcol = newcol - 1;
						end						

						if (r == newrow && c == newcol)
							break;
						end

						[keyloc keyval] = fitQuadratic(I_dog, i, newrow, newcol);							
					end

					% discard if | keyval | < 0.03
					if (abs(keyval) < 0.03)
						continue;	
					end

					% if (abs(keyloc(1)) > 1.5 || abs(keyloc(2)) > 1.5 || ...
					%      abs(keyloc(3)) > 1.5)
					% 	continue;
					% end

					% check if already created keypoint
					if (bitmap(newrow, newcol))
						continue;
					else
						bitmap(newrow, newcol) = 1;
					end				

					% determine scale of this keypoint, which is needed for computing the 
					% sigma of the gaussian weight-window 
					kpScale = initSigma * ( 2 ^ ((i + keyloc(1)) / intSize ) );

					% compute local histogram of gradient orientations and use it to 
					% build keypoint descriptors
					keys = orientationHist(gradMag, gradOrt, kpScale, ... 
						newrow + keyloc(2), newcol + keyloc(3), zoomLevel);
					keypoints = [keypoints; keys];

				end
			end
		end

	end
	fprintf('Got %d\n', count);