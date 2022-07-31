% Magnify an image to 2x
function I_exp = doubleimage(I)

	[M, N] = size(I);

	I_exp = zeros(2 * M, 2 * N);

	I_exp(1:2:end, 1:2:end) = I;

	I_exp(2:2:end-1, 1:2:end) = ( I(1:end-1, :) + I(2:end, :) ) / 2;

	I_exp(1:2:end, 2:2:end-1) = ( I(:, 1:end-1) + I(:, 2:end) ) / 2;

	I_exp(2:2:end-1, 2:2:end-1) = ...
		( I(1:end-1, 1:end-1)   + ...
		  I(2:end, 1:end-1)     + ...
		  I(1:end-1, 2:end)     + ...
		  I(2:end, 2:end) ) / 4;