% Given a 3d DoG function, return all extrema points
function maxima = getMaxima(I_dog)

	[M N S] = size(I_dog);
	maxima = [];

	for s = 2:S-1
		for i = 2:M-1
			for j = 2:N-1
				
				val = I_dog(i, j, s);
				ok = true;
				for z = -1:1
					for x = -1:1
						for y = -1:1
							if (y == 0 && x == 0 && z == 0) continue; end
							if (I_dog(i+x,j+y,s+z) >= val) ok = false; end
						end
						if (~ok) break; end
					end
					if (~ok) break; end
				end
				if (ok)
					maxima = [maxima, (s - 1) * (M * N) + (j - 1) * M + i];
				end

			end
		end
	end