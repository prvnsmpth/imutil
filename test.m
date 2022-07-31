
% test SIFT implementation
options.initSigma = 1.6;
options.intSize = 3;
options.showDogs = false;

% read first image 
fprintf('Computing keypoints for image 1...\n');
I1 = imread('img/adam1_s.png');
if (isrgb(I1))
	I1 = rgb2gray(I1);
end
I1 = double(I1);
keypoints1 = computeSift(I1, options);

% read second image
fprintf('Done.\nComputing keypoints for image 2...\n');
I2 = imread('img/adam2_s.png');
if (isrgb(I2))	
	I2 = rgb2gray(I2);
end
I2 = double(I2);
keypoints2 = computeSift(I2, options);

% show matching keypoints
fprintf('Done.\nComputing matches...\n');
matches = computeMatches(keypoints1, keypoints2);
fprintf('%d matches found', size(matches, 1));

% display keypoints
figure;
imshow(uint8(I1));
hold on;
plot(keypoints1(:, 1), keypoints1(:, 2), 'r+');

% display keypoints
figure;
imshow(uint8(I2));
hold on;
plot(keypoints2(:, 1), keypoints2(:, 2), 'r+');

showMatches(I1, I2, matches);
