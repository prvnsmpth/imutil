function dummy = showMatches(I1, I2, matches)

	[M N] = size(I1);
	[A B] = size(matches);
	MI = [I1, zeros(M, 10), I2];
	
	figure;
	imshow(uint8(MI)); hold on;
	for i = 1:A
		p1 = matches(i, 1:2);
		p2 = matches(i, 133:134);
		line([p1(1); p2(1) + 10 + N], [p1(2); p2(2)]);
	end