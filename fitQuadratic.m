% fit a 3D quadratic function at the location of the
% candidate keypoint detected by 26-neighbour comparison
% Given
%	l : the number of the layer in which the extrema lies
%	x : row number of extrema
%   y : col number of extrema
% Returns
%	accurately detected keypoint location
function [loc, val] = fitQuadratic(I_dog, l, x, y)

	% the three layers of interest
	A = I_dog(:, :, l - 1); % layer below
	B = I_dog(:, :, l); % same layer
	C = I_dog(:, :, l + 1); % layer above

	% compute first derivatives
	F = zeros(3, 1);
	F(1) = (C(x, y) - A(x, y)) / 2;				% derivate in Z direction
	F(2) = (B(x + 1, y) - B(x - 1, y)) / 2;		% derivate in X direction
	F(3) = (B(x, y + 1) - B(x, y - 1)) / 2;		% derivate in Y direction	

	% compute second derivatives
	S = zeros(3, 3);
	S(1, 1) = A(x, y) - 2 * B(x, y) + C(x, y);
	S(2, 2) = B(x + 1, y) - 2 * B(x, y) + B(x - 1, y);
	S(3, 3) = B(x, y + 1) - 2 * B(x, y) + B(x, y - 1);

	S(1, 2) = ( (C(x + 1, y) - C(x - 1, y)) - ...
					(A(x + 1, y) - A(x - 1, y)) ) / 4;
	S(2, 1) = S(1, 2);
	S(1, 3) = ( (C(x, y + 1) - C(x, y - 1)) - ...
					(A(x, y + 1) - A(x, y - 1)) ) / 4;
	S(3, 1) = S(1, 3);
	S(2, 3) = ((B(x + 1, y + 1) - B(x + 1,y - 1)) - ...
					(B(x - 1, y + 1) - B(x - 1, y - 1))) / 4;
	S(3, 2) = S(2, 3);

	% to get the location of the extremum, we need to solve
	% S * x = -F. Therefore, x = -inv(S) * F
	if (det(S) > 1e-8)
		loc = -1 * inv(S) * F;
	else
		loc = [0; 0; 0];
	end	

	% the value of the function at the extremum
	val = B(x, y) + 0.5 * (loc .* F);

