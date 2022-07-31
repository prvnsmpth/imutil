% Given a point on a layer of a DoG image, determines
% whether to accept or reject it based on the ratio of
% principal curvatures
function accept = checkEdgeResponse(dog, x, y)

	% compute 2D Hessian
	H = zeros(2, 2);
	H(1, 1) = dog(x + 1, y) - 2 * dog(x, y) + dog(x - 1, y);
	H(2, 2) = dog(x, y + 1) - 2 * dog(x, y) + dog(x, y - 1);
	H(1, 2) = ((dog(x + 1, y + 1) - dog(x + 1, y - 1)) ...
		- (dog(x - 1, y + 1) - dog(x - 1, y - 1))) / 4;
	H(2, 1) = H(1, 2);

	% compute trace and determinant
	tr = trace(H); de = det(H);

	% accept if principal curvatures ratio is less than 10
	accept = (tr * tr / (de + 1e-12)) < 12.1;