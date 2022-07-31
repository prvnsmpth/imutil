function offset = fitParabola(a, b, c)

	% if middle bar is negative, negate all three
	if (b < 0)
		a = -a;
		b = -b;
		c = -c;
	end

	% return offset from middle
	offset = 0.5 * (a - c) / (a - 2 * b + c); 