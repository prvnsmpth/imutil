function is = isMinima(val, vec)
	is = true;
	for i = 1:size(vec, 1)
		if (val => vec(i))
			is = false;
			break;
		end
	end