%&%% DIGITAL IMAGING PROCESSING
% Karam Mawas	    2946939 	

clear all
close all
clc


image1 = imread('C:\Users\KemOo_000\Documents\MATLAB\Third Semester\Image Based\Buildings\Gebaeude_0004_half.jpg');
image2 = imread('C:\Users\KemOo_000\Documents\MATLAB\Third Semester\Image Based\Buildings\Gebaeude_0005_half.jpg');

image1 = imresize(image1, 0.25);
image2 = imresize(image2, 0.25);

imshow(image1)

title('digitize new points image1')
[x,y]=getline('close');

% getline provides polygon with identical first and last points
NPts = length(x)-1;
if NPts < 4
    fpritf('Number of Measured Point Should Be at least 4');
end

% build a matrix for the selected points
pts1 = [x(1:NPts) y(1:NPts)]';
hold on
% Marking the points
plot([x(1:NPts);x(1)],[y(1:NPts);y(1)],'-ro')
title ('digitize of points succeeded')
drawnow


% get four points from image2 by just clicking a rectangle
f=figure(1);
imshow(image2)
% measure manually
title('digitize the points in image 2')
[x,y] = getline('close');
if [length(x)-1] ~= NPts
    fprintf('Number of Measured Points in Left and right images should be equal');
    figure(1); hold on
    imshow(image2)
    title('Error occurred Number of points in both images should be equal','color','R')
    
else
pts2 = [x(1:NPts) y(1:NPts)]';
hold on
plot([x(1:NPts);x(1)],[y(1:NPts);y(1)],'-ro')
title ('digitize of points succeeded')
drawnow
end
fprintf('Number of Measured Points %i\n',NPts);

% generate gray value images - to be used for rectification

im1gray = rgb2gray(image1);
im2gray = rgb2gray(image2);

% generate homogenous coordinates
x1 = [pts1 ; ones(1,NPts)];
x2 = [pts2 ; ones(1,NPts)];

% calculate the perspetive transformation
% kind of function
% T = homography2d([pts1 ; ones(1,NPts) ; pts2 ; ones(1,NPts)]);
% A = zeros(2*NPts,9);
% 
% o = [0 0 0];
% to solve a linear problem in SVD
A = zeros(2*NPts,9);
o = [0 0 0];
for n = 1:NPts
    X = x1(:,n)';
    x = x2(1,n);
    y = x2(2,n);
    w = x2(3,n);
    % building the A matrix
    A(2*n-1,:) = [ o -w*X y*X];
    A(2*n,:) = [ w*X o x*X];
end
% SVD
[U,D,V] = svd(A,0);
% Extract Homography
H = reshape(V(:,9),3,3)';

% test: Transform measured points using homography H
x_calc = H*x1;

% Normalize homogeneous coordinates to a scale of 1
xcalc(1,:) = x_calc(1,:)./x_calc(3,:);
xcalc(2,:) = x_calc(2,:)./x_calc(3,:);
xcalc(3,:) = x_calc(3,:)./x_calc(3,:);

% compute differences between transformed and original points for control
diff = x_calc - x2;
for i = 1:NPts
    fprintf('point %i: Differences x2-xcalc [X,Y] %6.3f %6.3f \n',i,diff(1:2,i));
end

% Now transform the complete image
% just use Pan image or (channel 1) to speed up the process
[width height numbands] = size(image1);
%generate Matix with pixel coordinates
[xi,yi] = meshgrid(1:height, 1:width);  % All possible xy coord. in the image
TransPoints = [xi(:) yi(:) ones(length(yi(:)),1)]';
% Multiply coordinates with homography,
pix_points = H * TransPoints;
% Normalization to determine pixel coordinates
pix_points(1,:) = (pix_points(1,:) ./ pix_points(3,:));
pix_points(2,:) = (pix_points(2,:) ./ pix_points(3,:));
pix_points(3,:) = [];
% Generate image array
xi = reshape(pix_points(1,:),width,height);
yi = reshape(pix_points(2,:),width,height);
% Interpolation from Pan image to speed up the process
newim = interp2(double(im2gray),xi,yi,'linear',0);
newim=mat2gray(newim);
% imshow(newim,[])
imtool(newim);
title('Rectified')
% imwrite(newim,'rectif_img.jpg','JPEG');
