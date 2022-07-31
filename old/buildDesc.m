% Sample keypoints in a circular neighbourhood of given point and build
% the keypoint descriptor vector
%
% Given
% 	grad : M x N matrix of gradient magnitudes
% 	ort : M x N matrix of gradient orientations
%	kpScale : scale of layer containing the point
%	x : row number of point
%	y : col number of point
% 	ortAng : The orientation angle of this point
%	zoomLevel : down-sampling factor
% Return
% 	desc : The descriptor vector for this keypoint
%
%
% import io
% import sys
% from PIL import Image
%
% def _embed(container_matrix, source_matrix, source_w, source_h):
%   for i in range(source_w):
%     for j in range(source_h):
%       val = source_matrix[i, j]
%       a = val & 0x03
%       b = (val & 0x0c) >> 2
%       c = (val & 0x30) >> 4
%       d = (val & 0xc0) >> 6
%       arr = [d, c, b, a]
%       for k in range(4):
%         data = arr[k]
%         c_i = i * 2 + k // 2
%         c_j = j * 2 + k % 2
%         c_r, c_g, c_b = container_matrix[c_i, c_j]
%         first = data >> 1
%         second = data & 0x01
%         c_r = c_r | 0x01 if first > 0 else ((c_r // 2) * 2)
%         c_g = c_g | 0x01 if second > 0 else ((c_g // 2) * 2)
%         container_matrix[c_i, c_j] = (c_r, c_g, c_b)
%
% def _extract(merged_matrix, source_matrix, source_w, source_h):
%   for i in range(source_w):
%     for j in range(source_h):
%       val = 0
%
%       for k in range(4):
%         m_i, m_j = i * 2 + k // 2, j * 2 + k % 2
%         r, g, _ = merged_matrix[m_i, m_j]
%         data = ((r & 0x01) << 1) + (g & 0x01)
%         val = val << 2
%         val += data
%
%       source_matrix[i, j] = int(val)
%
% # The source_file is written to the container_file
% def merge(container_file, source_file):
%   container_img = Image.open(container_file)
%   source_img = Image.open(source_file).convert('L') # Convert to grayscale
%
%   container_w, container_h = container_img.size
%   source_w, source_h = source_img.size
%
%   # Resize container if needed
%   if source_w * 2 != container_w or source_h * 2 != container_h:
%     print('Resizing image')
%     container_img = container_img.resize((source_w * 2, source_h * 2))
%
%   container_pixels = container_img.load()
%   source_pixels = source_img.load()
%   _embed(container_pixels, source_pixels, source_w, source_h)
%   container_img.save('merged.png')
%
%
% def unmerge(merged_file):
%   merged_img = Image.open(merged_file)
%   merged_w, merged_h = merged_img.size
%
%   source_w, source_h = merged_w // 2, merged_h // 2
%   source_img = Image.new('L', (source_w, source_h))
%
%   merged_pixels = merged_img.load()
%   source_pixels = source_img.load()
%   _extract(merged_pixels, source_pixels, source_w, source_h)
%   source_img.save('output.png')
%
% if __name__ == '__main__':
%   cmd = sys.argv[1]
%   if cmd != 'merge' and cmd != 'unmerge':
%     print('Usage: merge.py <merge|unmerge> [container_file] [source_file]')
%     exit(1)
%
%   container_file = sys.argv[2]
%   if cmd == 'merge':
%     source_file = sys.argv[3]
%     merge(container_file, source_file)
%   elif cmd == 'unmerge':
%     unmerge(container_file)
%
function desc = buildDesc(grad, ort, kpScale, x, y, ortAng, zoomLevel)

	[M N] = size(grad);

	% These go into the final descriptor vector
	orig_x = zoomLevel * x; 		% keypoint position in original image
	orig_y = zoomLevel * y;
	scl = zoomLevel * kpScale; 			% scale relative to first image in first octave

	% we are building a 4x4 array of orientation histograms
	sampleSize = 4;
	num_orients = 8;
	windowWid = 3 * kpScale;

	% histogram for local descriptor
	histogram = zeros(sampleSize, sampleSize, num_orients);

	% compute radius of the circle
	radius = round(( sqrt(2) * (sampleSize + 1) * (windowWid) ) / 2);
	sig = 0.5 * sampleSize;

	% iterate through points that can lie inside the 16x16 region
	for i = -radius:radius
		for j = -radius:radius

			% check if the point lies inside the image boundaries
			if (x + i >= 1 && x + i <= M && ...
				y + j >= 1 && y + j <= N)

				% rotate offset (i, j) to be relative to orientation of the keypoint
				temp = (1 / (windowWid)) * ...
							( [cos(ortAng) -sin(ortAng); ...
							   sin(ortAng) cos(ortAng)] * [i; j] ) + ...
								[ sampleSize / 2 + 1; sampleSize / 2 + 1 ];
				drow = temp(1); dcol = temp(2);

				% check if (drow, dcol) is a valid index into the descriptor array (4x4)
				if (drow > 0 && drow < (sampleSize + 1) && ...
					dcol > 0 && dcol < (sampleSize + 1))

					% compute the weight of this sample point
					temp = [drow; dcol] - [sampleSize / 2 + 1; sampleSize / 2 + 1];
					offr = temp(1); offc = temp(2);
					w = exp( -(offr * offr + offc * offc) / (2 * sig * sig) );
					gradVal = grad(round(x + i), round(y + j));

					% weighted value to be added into histogram
					wv = w * gradVal;

					% compute orientation w.r.t. orientation of keypoint
					ortVal = ort(round(x + i), round(y + j)) - ortAng;
					ortVal = rem(ortVal, 2 * pi);
					if (ortVal < 0)
						ortVal = ortVal + 2 * pi;
					end

					% what direction?
					dirct = num_orients * ( ortVal / (2 * pi) );

					% round off the indexes
					drow_r = round(drow);
					dcol_r = round(dcol);
					dirct_r = round(dirct);
					drow_f = drow - drow_r;
					dcol_f = dcol - dcol_r;
					dirct_f = dirct - dirct_r;

					% now update the appropriate histograms
					for it = 0:1
						ri = drow_r + it;
						if (ri >= 1 && ri <= sampleSize)
							rowt = gradVal * ite(it == 0, 1 - drow_f, drow_f);
							for jt = 0:1
								ci = dcol_r + jt;
								colwt = rowt * ite(jt == 0, 1 - dcol_f, dcol_f);
								if (ci >= 1 && ci <= sampleSize)
									for kt = 0:1
										oi = dirct_r + kt;
										if (oi > num_orients)
											oi = 1;
										end
										ortwt = colwt * ite(kt == 0, 1 - dirct_f, dirct_f);
										if (oi >= 1 && oi <= num_orients)
											histogram(ri, ci, oi) = ...
												histogram(ri, ci, oi) + ortwt;
										end
									end
								end

							end
						end

					end

				end

			end

		end
	end

	% unroll the 3d histogram into a vector
	desc = reshape(histogram, 1, sampleSize * sampleSize * num_orients);
	desc = ( 1 / norm(desc) ) * desc;

	% limit values in the vector at 0.2
	desc = min(desc, 0.2);
	desc = ( 1 / norm(desc) ) * desc;

	% add the x and y position, scale and orientation
	desc = [orig_x, orig_y, scl, ortAng, desc];
	fprintf('Keypoint added: (%d, %d)\n', orig_x, orig_y);

