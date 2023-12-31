% demo_cartesian_recon_SE_axial.m
% Written by Nam Gyun Lee
% Email: namgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 01/16/2022, Last modified: 01/16/2022

%% Clean slate
close all; clear all; clc;

%% Set source directories
src_directory = '/server/home/ncan/GitHub/lowfield_maxgirf';
ismrmrd_directory = '/server/home/nlee/ismrmrd';

%% Add source directories to search path
addpath(genpath(src_directory));
addpath(genpath(ismrmrd_directory));

%% Define data directory
ismrmrd_noise_fullpath = '/server/sdata/ncan/NHLBI/20201102_NV_brain/noise/noise_meas_MID00273_FID03656_se_15b130_tra.h5';
ismrmrd_data_fullpath  = '/server/sdata/ncan/NHLBI/20201102_NV_brain/h5/meas_MID00273_FID03656_se_15b130_tra.h5';
siemens_dat_fullpath   = '/server/sdata/ncan/NHLBI/20201102_NV_brain/meas_MID00273_FID03656_se_15b130_tra.dat';

% "phase_sign" and "read_sign" can be determined only from Siemens raw data 
% format now until the ISMRMRD format includes these as part of its header
user_opts.phase_sign =  1;
user_opts.read_sign  = -1;
user_opts.remove_oversampling = 1; % Remove readout oversampling

%% Perform FFT reconstruction
[im_se,header,r_dcs] = siemens_cartesian_fft_recon_multislice(ismrmrd_noise_fullpath, ismrmrd_data_fullpath, siemens_dat_fullpath, user_opts);
[N1,N2,Ns] = size(im_se);

%% Save results
[filepath,filename,ext] = fileparts(siemens_dat_fullpath);
save_filename = sprintf('%s_ro%d', filename, user_opts.remove_oversampling);
save(save_filename, 'im_se', 'r_dcs', '-v7.3');

%% Display images in DCS
figure('Color', 'w'); hold on;
for idx = 1:Ns
    X = reshape(r_dcs(:,1,idx), [N1 N2]);
    Y = reshape(r_dcs(:,2,idx), [N1 N2]);
    Z = reshape(r_dcs(:,3,idx), [N1 N2]);
    surf(X*1e3, Y*1e3, Z*1e3, abs(im_se(:,:,idx)), 'EdgeColor', 'none');
end
axis image tight;
colormap(gray(256));
caxis([0 12]);
xlabel('x [mm]');
ylabel('y [mm]');
zlabel('z [mm]');
title('Physical Coordinate System');
set(gca, 'ZDir', 'reverse');
set(gca, 'XDir', 'reverse');
view(-123,25);

%% Display images
reorient = @(x) x.';
[filepath,filename,ext] = fileparts(ismrmrd_data_fullpath);

for idx = 1:Ns
    slice_nr = idx;
    %----------------------------------------------------------------------
    % Display a SE image
    %----------------------------------------------------------------------
    figure('Color', 'k', 'Position', [4 201 1553 597]);
    color_order = get(gca, 'colororder');
    imagesc(reshape(abs(reorient(im_se(:,:,slice_nr))), [N1 N2])); axis image;
    set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    caxis([0 12]);
    colormap(gray(256));
    hc = colorbar;
    set(hc, 'Color', 'w', 'FontSize', 14);
    text(N2 / 2, 0, {'Magnitude of Cartesian SE'}, 'FontSize', 14, 'Color', color_order(3,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    export_fig(sprintf('%s_slice%d_mag', filename, slice_nr), '-r300', '-tif', '-c[70,1430,160,1620]'); % [top,right,bottom,left]
    close gcf;

    figure('Color', 'k', 'Position', [4 201 1553 597]);
    color_order = get(gca, 'colororder');
    imagesc(reshape(angle(reorient(im_se(:,:,slice_nr)))*180/pi, [N1 N2])); axis image;
    set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    colormap(hsv(256));
    hc = colorbar;
    set(hc, 'Color', 'w', 'FontSize', 14);
    text(N2 / 2, 0, {'Phase of Cartesian SE'}, 'FontSize', 14, 'Color', color_order(3,:), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    export_fig(sprintf('%s_slice%d_phase', filename, slice_nr), '-r300', '-tif', '-c[70,1430,160,1620]'); % [top,right,bottom,left]
    close gcf;
end