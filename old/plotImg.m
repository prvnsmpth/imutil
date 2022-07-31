% Given a cell array of images from octaves, plots on single figure
function dummy = plotImg(im)

	% M = # of octaves, N = 1
	[M N] = size(im);

	% num images per row
	rowLen = ite(rem(size(im{1}, 3), 5) == 0, 5, 4);

	% iterate over octaves and plot
	figure;
	for i = 1:M
		[A B C] = size(im{i});
		for j = 1:C
			subplot(M, rowLen, (i - 1) * rowLen + j);
			I = im{i}(:, :, j);
			imshow(I / max(I(:)));
		end
	end


	