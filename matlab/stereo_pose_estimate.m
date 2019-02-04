%%
%mklink /J "d:\DATA\EON1" "E:\IT Quality Services Kereskedelmi és Szolgáltató Kft\Kutatás-Fejlesztés - Dokumentumok\OneDrive - IT Quality Services Kereskedelmi és Szolgáltató Kft\EON"
%%
%run stereo calibration app and export stereoParams
showExtrinsics(stereoParams);
%%
base_dir='d:\DATA\MAV1\Images\';
cur_id='Selection_1\';
frame_num=12700;
im_1=fullfile(base_dir,cur_id,['roi_0_',num2str(frame_num),'.jpg']);
im_2=fullfile(base_dir,cur_id,['roi_1_',num2str(frame_num),'.jpg']);


I1  = imread(im_1);
I2  = imread(im_2);

[frameLeftRect, frameRightRect] = ...
    rectifyStereoImages(I1, I2, stereoParams);

%The disparity range depends on the distance between the two cameras and the distance between the cameras and the object of interest. Increase the DisparityRange when the cameras are far apart or the objects are close to the cameras. To determine a reasonable disparity for your configuration, display the stereo anaglyph of the input images in imtool and use the Distance tool to measure distances between pairs of corresponding points. Modify the MaxDisparity to correspond to the measurement.
figure;
imtool(stereoAnaglyph(frameLeftRect, frameRightRect));

%frameLeft_undistorted = undistortImage(I1,stereoParams.CameraParameters1);
%frameRight_undistorted = undistortImage(I2,stereoParams.CameraParameters2);

frameLeftGray  = rgb2gray(frameLeftRect);
frameRightGray = rgb2gray(frameRightRect);

%frameLeftGray  = rgb2gray(frameLeft_undistorted);
%frameRightGray = rgb2gray(frameRight_undistorted);

disparityRange = [0 64];
disparityMap = disparity(frameLeftGray,frameRightGray,'BlockSize',...
    11,'DisparityRange',disparityRange);figure;
imshow(disparityMap, disparityRange);
title('Disparity Map');
colormap jet
colorbar

%%
% distance measure
f1=figure;
imshow(I1);
rect1=getrect(f1);
f2=figure;
imshow(I2);
rect2=getrect(f2);


point3d = triangulate(rect1(1:2), rect2(1:2), stereoParams);
distanceInMeters = norm(point3d)/1000;

figure;
imshow(insertObjectAnnotation(frameRightRect, 'rectangle', rect2, [num2str(distanceInMeters),' meter'],'FontSize',72,'LineWidth',10));
title('Detected Pylon');
%%
points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

% Create a streaming point cloud viewer
player3D = pcplayer([-30, 3], [-30, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);