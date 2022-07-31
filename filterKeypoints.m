% Filters out keypoints that lie on edges, and accurately
% locates keypoints using a quadratic fit model
function filt_keys = filterKeypoints(keys, I_dog)

	[M N S] = size(I_dog);
	[R C] = size(keys);

	filt_keys = zeros(3, 0);
	for i = 1:C
		
		% co-ordinates of keypoint
		col = keys(1, i) + 1;
		row = keys(2, i) + 1;
		lay = keys(3, i) + 2;
		dog = I_dog(:, :, lay);

		% check edge response to determine if point lies on edge, if yes,
		% then discard
		if (checkEdgeResponse(dog, row, col))

			% quadratic fit to determine location accurately
			[offset val] = fitQuadratic(I_dog, lay, row, col);			
			maxMoves = 5; newrow = row; newcol = col;
			while (1)
				if (maxMoves == 0)
					break;
				end
				maxMoves = maxMoves - 1;
				
				r = newrow; c = newcol;
				if ( offset(2) > 0.5 && newrow < M - 2 )
					newrow = newrow + 1;
				elseif ( offset(2) < -0.5 && newrow > 3 )
					newrow = newrow - 1;
				end

				if ( offset(3) > 0.5 && newcol < N - 2 )
					newcol = newcol + 1;
				elseif ( offset(3) < -0.5 && newcol > 3 )
					newcol = newcol - 1;
				end						

				if (r == newrow && c == newcol)
					break;
				end

				[offset val] = fitQuadratic(I_dog, lay, newrow, newcol);					
			end

			% the point has been moved, not determine whether to include
			% or not
			if (abs(val) < 0.01)
				continue;
			end
				
			if (abs(offset(1)) > 1.5 || abs(offset(2)) > 1.5 || ... 
				abs(offset(3)) > 1.5)
				continue;
			end

			% add this point to list of filtered keypoints
			lay = round(lay + offset(1));
			row = round(newrow + offset(2));
			col = round(newcol + offset(3));			
			filt_keys = [filt_keys, [col - 1; row - 1; lay - 2]];
		end

	end