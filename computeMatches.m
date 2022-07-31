% Given keypoints of two images, determing the matching keypoints
function matches = computeMatches(list1, list2)

	[M1 N1] = size(list1);
	[M2 N2] = size(list2);
	matches = [];

	% try to find a match for every point in list1 in list2
	for i = 1:M1
		d1 = 1e99;
		d2 = d1;
		minIdx = -1;
		secMinIdx = minIdx;
		for j = 1:M2
			distnc = norm(list1(i, 5:end) - list2(j, 5:end));
			if (distnc < d1)
				d2 = d1;
				d1 = distnc;
				secMinIdx = minIdx;
				minIdx = j;
			elseif (distnc < d2)
				d2 = distnc;
				secMinIdx = j;
			end
		end

		% perform threshold check on the distance ratio
		if (d1 / d2 < 0.8)
			matches = [matches; list1(i, :), list2(minIdx, :)];
		end

	end