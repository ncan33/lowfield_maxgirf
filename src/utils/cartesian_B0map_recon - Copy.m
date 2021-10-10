%function [im_echo,Masks,B0maps,x_echo,y_echo,z_echo,TE,csm] = cartesian_B0map_recon(gre_noise_fullpaths, gre_data_fullpaths, gre_dat_fullpaths, user_opts_cartesian, save_figures)
% Inputs
%   gre_noise_full_paths    cell structure containing full paths to noise only h5 files
%   gre_data_full_paths     cell structure containing full paths to mutlti-echo h5 files
%   gre_dat_full_paths      cell structure containing full paths to Siemens dat files


%% Parse the user options
output_directory = user_opts_cartesian.output_directory;
zpad_factor      = user_opts_cartesian.zpad_factor;
main_orientation = user_opts_cartesian.main_orientation;
lambda           = user_opts_cartesian.lambda;
maxiter_tgv      = user_opts_cartesian.maxiter_tgv;

%% Define constants
gamma = 4257.59 * (1e4 * 2 * pi); % gyromagnetic ratio for 1H [rad/sec/T]

%% Read the first ismrmrd file to get header information
start_time = tic;
tic; fprintf('Reading an ismrmrd file: %s... ', gre_data_fullpaths{1});
if exist(gre_data_fullpaths{1}, 'file')
    dset = ismrmrd.Dataset(gre_data_fullpaths{1}, 'dataset');
    fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
else
    error('File %s does not exist.  Please generate it.' , gre_data_fullpaths{1});
end

%% Get imaging parameters from the XML header
header = ismrmrd.xml.deserialize(dset.readxml);

%--------------------------------------------------------------------------
% Sequence parameters
%--------------------------------------------------------------------------
TR         = header.sequenceParameters.TR * 1e-3;     % [msec] * [sec/1e3msec] => [sec]
TE         = header.sequenceParameters.TE * 1e-3;     % [msec] * [sec/1e3msec] => [sec]
flip_angle = header.sequenceParameters.flipAngle_deg; % [degrees]

%--------------------------------------------------------------------------
% Experimental conditions
%--------------------------------------------------------------------------
B0 = header.experimentalConditions.H1resonanceFrequency_Hz * (2 * pi / gamma); % [Hz] * [2pi rad/cycle] / [rad/sec/T] => [T]

%% Parse the ISMRMRD header
raw_data = dset.readAcquisition(); % read all the acquisitions

%% Get imaging parameters from the XML header
header = ismrmrd.xml.deserialize(dset.readxml);

%% Set the dimension of the data
N1 = zpad_factor * header.encoding.reconSpace.matrixSize.x; % number of voxels in image-space (RO,row)
N2 = zpad_factor * header.encoding.reconSpace.matrixSize.y; % number of voxels in image-space (PE,col)
N3 = double(max(raw_data.head.idx.slice)) + 1; % number of slices
Ne = length(gre_data_fullpaths); % number of echoes
N  = N1 * N2; % total number of voxels in image-space

%% Reconstruct multi-echo GRE data
im_echo = complex(zeros(N1, N2, N3, Ne, 'double'));
TEs = zeros(Ne, 1, 'double'); % [sec]

for idx = 1:Ne
    %----------------------------------------------------------------------
    % Perform Cartesian reconstruction
    %----------------------------------------------------------------------
    if idx == 1
        [im,header,x,y,z,imc,csm] = cartesian_recon(gre_noise_fullpaths{idx}, gre_data_fullpaths{idx}, gre_dat_fullpaths{idx}, user_opts_cartesian);
        csm = squeeze(csm); % N1 x N2 x N3 x Nc
    else
        [~,header,x,y,z,imc] = cartesian_recon(gre_noise_fullpaths{idx}, gre_data_fullpaths{idx}, gre_dat_fullpaths{idx}, user_opts_cartesian);
        imc = squeeze(imc); % N1 x N2 x N3 x Nc
        im = sum(conj(csm) .* imc, ndims(imc)) ./ sum(abs(csm).^2, ndims(imc));
    end

    %----------------------------------------------------------------------
    % Save the reconstructed image
    %----------------------------------------------------------------------
    im_echo(:,:,:,idx) = reshape(im, [N1 N2 N3]);
    TEs(idx) = header.sequenceParameters.TE * 1e-3; % [msec] * [sec/1e3msec] => [sec]
end

return

%% Get spatial coordinates in DCS [m]
x_echo = reshape(x, [N1 N2 N3]); % N1 x N2 x N3
y_echo = reshape(y, [N1 N2 N3]); % N1 x N2 x N3
z_echo = reshape(z, [N1 N2 N3]); % N1 x N2 x N3

%% Calculate a raw static off-resonance map [Hz]
B0map_raw = zeros(N1, N2, N3, 'double'); % [Hz]
for idx3 = 1:N3
    tic; fprintf('(%2d/%2d): Calculating a static off-resonance map... ', idx3, N3);
    nr_voxels = N1 * N2;
    for idx = 1:nr_voxels
        %------------------------------------------------------------------
        % Perform least-squares fitting per voxel
        % m = a * TE + b, m in [rad], a in [rad/sec], TE in [sec]
        % m(1) = a * TE(1) + b    [m(1)] = [TE(1) 1][a]
        % m(2) = a * TE(2) + b => [m(2)] = [TE(2) 1][b] => m = Av
        % m(N) = a * TE(N) + b    [m(N)] = [TE(N) 1]
        %------------------------------------------------------------------
        [idx1,idx2] = ind2sub([N1 N2], idx);
        m = unwrap(reshape(angle(im_echo(idx1,idx2,idx3,:)), [Ne 1])); % [rad]
        A = [TEs ones(Ne,1)]; % [sec]
        v = A \ m; % [rad/sec]
        B0map_raw(idx1,idx2,idx3) = v(1) / (2 * pi); % [rad/sec] * [cycle/2pi rad] => [Hz]
    end
    fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
end

% %%
% % HZ per ppm
% HZPPPM = gamma * B0 / (2 * pi) * 1e-6; % [HZ/ppm]
% water_ppm = 4.65;
% fat_ppm = 1.3;
% water_fat_difference_Hz = (water_ppm - fat_ppm) * HZPPPM;
% 
% tau = 1 / ((water_ppm - fat_ppm) * HZPPPM) * 1e3; % [msec]
% 
% %gamma = 4257.59 * (1e4 * 2 * pi); % gyromagnetic ratio for 1H [rad/sec/T]
% 
% %%
% figure('Color', 'w');
% montage(reorient(B0map_raw), 'DisplayRange', []);
% colormap(hot(256)); colorbar;
% caxis([-90 90]);
% title('Raw static off-resonance map [Hz]')
% impixelinfo;
% 
% %%
% figure('Color', 'w');
% montage(reorient(abs(im_echo(:,:,:,1))), 'DisplayRange', []);
% colormap(gray(256));
% 
% return

%% Calculate a mask defining the image and noise regions using the iterative intermean algorithm
%--------------------------------------------------------------------------
% Calculate the maximum magnitude of images from all echo points
%--------------------------------------------------------------------------
im_max = max(abs(im_echo), [], 4);

%--------------------------------------------------------------------------
% Calculate a mask
%--------------------------------------------------------------------------
mask = false(N1, N2, N3);
for idx = 1:N3
    level = isodata(im_max(:,:,idx), 'log');
    mask(:,:,idx) = (im_max(:,:,idx) > level);
end

%% Perform least-squares fitting using a water-fat signal model
f_F = -3.5 * gamma * B0 / (2 * pi) * 1e-6; % [ppm] * [Hz/ppm] => [Hz]

m_w = complex(zeros(N1, N2, N3, 'double')); % [Hz]
m_f = complex(zeros(N1, N2, N3, 'double')); % [Hz]
df0  = zeros(N1, N2, N3, 'double'); % [Hz]

m_w2 = complex(zeros(N1, N2, N3, 'double')); % [Hz]
m_f2 = complex(zeros(N1, N2, N3, 'double')); % [Hz]
df02 = zeros(N1, N2, N3, 'double'); % [Hz]


%options = optimset('display', 'iter');
options = optimset('Display', 'off', 'TolFun', 1e-12, 'TolX', 1e-12);

start_time = tic;
for idx3 = 6%:N3
    voxel_list = find(mask(:,:,idx3));
    nr_voxels = length(voxel_list);
    for idx = 1:nr_voxels
        % real variables (VARPRO)
        tic; fprintf('(%2d/%2d): Performing water-fat signal fitting (%3d/%3d)... ', idx3, N3, idx, nr_voxels);
        [idx1,idx2] = ind2sub([N1 N2], voxel_list(idx));
        y_complex = reshape(im_echo(idx1,idx2,idx3,:), [Ne 1]);
        y = cat(1, real(y_complex), imag(y_complex));
        alphainit = 0;
        w = ones(2*Ne, 1, 'double');
        [alpha,c,wresid,resid_norm,y_est,Regression] = ...
            varpro(y,w,alphainit,4,@(alpha)eta_water_fat_signal_model_real(alpha,f_F,TEs),[],[],options);
        m_w2(idx1,idx2,idx3) = complex(c(1), c(3));
        m_f2(idx1,idx2,idx3) = complex(c(2), c(4));
        df02(idx1,idx2,idx3) = alpha;
        fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
        
        
        
%         % Complex variables (VARPRO)
%         tic; fprintf('(%2d/%2d): Performing water-fat signal fitting (%3d/%3d)... ', idx3, N3, idx, nr_voxels);
%         [idx1,idx2] = ind2sub([N1 N2], voxel_list(idx));
%         y = reshape(im_echo(idx1,idx2,idx3,:), [Ne 1]);
%         alphainit = 0;
%         w = ones(Ne, 1, 'double');
%         [alpha,c,wresid,resid_norm,y_est,Regression] = ...
%             varpro(y,w,alphainit,2,@(alpha)eta_water_fat_signal_model_complex(alpha,f_F,TEs),[],[],options);
%         m_w(idx1,idx2,idx3) = c(1);
%         m_f(idx1,idx2,idx3) = c(2);
%         df0(idx1,idx2,idx3) = alpha;
%         fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
        
        
        
        
%         y_max = max(abs(y));
%         scale_factor = 100 / y_max;
%         y_scaled = y * scale_factor;

%         if 0
%         tic; fprintf('(%2d/%2d): Performing water-fat signal fitting (%3d/%3d)... ', idx3, N3, idx, nr_voxels);
%         %------------------------------------------------------------------
%         % Perform least-squares fitting per voxel
%         % s_1 = (m_w + m_f * exp(1j * 2 * pi * df1 * TE_1)) * exp(1j * 2 * pi * df0 * TE_1)
%         % s_2 = (m_w + m_f * exp(1j * 2 * pi * df1 * TE_1)) * exp(1j * 2 * pi * df0 * TE_2)
%         % s_N = (m_w + m_f * exp(1j * 2 * pi * df1 * TE_1)) * exp(1j * 2 * pi * df0 * TE_N)
%         %
%         % [s_1] = [exp(1j * 2 * pi * df0 * TE_1) exp(1j * 2 * pi * (df1 + df0) * TE_1)][m_w]
%         % [s_2] = [exp(1j * 2 * pi * df0 * TE_2) exp(1j * 2 * pi * (df1 + df0) * TE_2)][m_f]
%         % [s_N] = [exp(1j * 2 * pi * df0 * TE_N) exp(1j * 2 * pi * (df1 + df0) * TE_N)]
%         %
%         % y = Phi(alpha) * c
%         % y in C^(N x 1), Phi(alpha) in C^(N x 2), and c in C^(2 x 1)
%         %------------------------------------------------------------------
%         x0 = [0; 0; 0; 0; 0];
%         x = lsqnonlin(@(x)water_fat_signal_model(x,f_F,TEs,y), x0, [], [], options);
%         m_w(idx1,idx2,idx3) = complex(x(1), x(3));
%         m_f(idx1,idx2,idx3) = complex(x(2), x(4));
%         df0(idx1,idx2,idx3)  = x(5);
%         fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
% 
%         end
        
%         tic; fprintf('(%2d/%2d): Performing water-fat signal fitting (%3d/%3d)... ', idx3, N3, idx, nr_voxels);
% %         [idx1,idx2] = ind2sub([N1 N2], voxel_list(idx));
% %         y = reshape(im_echo(idx1,idx2,idx3,:), [Ne 1]);
%         yy = cat(1, real(y_scaled), imag(y_scaled));
%         alphainit = 0;
%         w = ones(2*Ne, 1, 'double');
%         [alpha,c,wresid,resid_norm,y_est,Regression] = ...
%             varpro(yy,w,alphainit,4,@(alpha)eta_water_fat_signal_model(alpha,f_F,TEs),[],[],options);
%         c = c / scale_factor;
%         m_w2(idx1,idx2,idx3) = complex(c(1), c(3));
%         m_f2(idx1,idx2,idx3) = complex(c(2), c(4));
%         df02(idx1,idx2,idx3) = alpha;
%         fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
%         cat(2, x, [c; alpha])
%         pause;
        
        
        
%         [Phi,dPhi,Ind] = eta_water_fat_signal_model(alpha, df1, TEs);
%         yy_fit = Phi * c; 
%         figure, hold on;
%         plot(yy);
%         plot(yy_noisy);
    end
end
%%


%%
return

res = water_fat_signal_model(x,f_F,TEs,y);

figure, montage(angle(reorient(im_echo(:,:,6,:))), 'DisplayRange', [], 'Size', [1 5]); colormap(hsv(256));
% %%
% %y = reshape(im_echo(idx1,idx2,idx3,:), [Ne 1]);
% x0 = [10; 0; 0; 20; 20];
% x0 = [10; 20; 10; 20; 40];
% df1 = 3.5 * gamma * B0 / (2 * pi) * 1e-6; % [ppm] * [Hz/ppm] => [Hz]
% yy = water_fat_signal_model(x0,df1,TEs,0);
% y = complex(yy(1:5), yy(6:end));
% 
% x = lsqnonlin(@(x)water_fat_signal_model(x,df1,TEs,y), x0*0, [], [], options);
% x

%%
x0 = [10; 20; 10; 20; 40];
f_F = -3.5 * gamma * B0 / (2 * pi) * 1e-6; % [ppm] * [Hz/ppm] => [Hz]
y = water_fat_signal_model(x0,f_F,TEs,0); % [real(y); imag(y)]
noise = 5 * randn(10,1);

y_noisy = y + noise;
y_complex = complex(y_noisy(1:5), y_noisy(6:end));

x1 = lsqnonlin(@(x)water_fat_signal_model(x,f_F,TEs,y_complex), x0*0, [], [], options);
%x1

alphainit = 0;
w = ones(2*Ne, 1, 'double');
[alpha,c,wresid,resid_norm,y_est,Regression] = ...
    varpro(y_noisy,w,alphainit,4,@(alpha)eta_water_fat_signal_model(alpha,f_F,TEs),[],[],options);

%x0
cat(2, x0, x1, [c; alpha])

[Phi,dPhi,Ind] = eta_water_fat_signal_model(alpha, f_F, TEs);
y_fit = Phi * c;

figure, hold on;
plot(y);
plot(y_noisy);
plot(y_fit);


%%
%%need to make a real problem1!!

%%
reorient = @(x) flip(rot90(x, -1), 2);

figure;
montage(reorient(mask));
%%

figure;
imagesc(abs(reorient(m_w2(:,:,6)))); axis image;
colormap(gray(256));
caxis([0 1e4]);
%%

figure;
imagesc(abs(reorient(m_f2(:,:,6)))); axis image;
colormap(gray(256));
caxis([0 1e4]);
impixelinfo
%%
figure;
imagesc((reorient((df02(:,:,6))))); axis image;
colormap(hot(256));
colorbar;
caxis([-60 60]);
%y = im_echo
impixelinfo

%%
figure;
imagesc((reorient(B0map_raw(:,:,6)))); axis image;
colormap(hot(256));
colorbar;
caxis([-60 60]);
%y = im_echo
impixelinfo



%% Calculate a smooth spherical harmonics approximation to the off-resonance map
B0map_fit = zeros(N1, N2, N3, 'double');

for idx3 = 1:N3
    %% Set spatial coordinates of the current slice
    x = reshape(x_echo(:,:,idx3), [N 1]);
    y = reshape(y_echo(:,:,idx3), [N 1]);
    z = reshape(z_echo(:,:,idx3), [N 1]);

    %% Calculate real-valued spherical harmonic basis functions
    tic; fprintf('(%2d/%2d): Calculating real-valued spherical harmonic basis functions... ', idx3, N3);
    %----------------------------------------------------------------------
    % Calculate the number of basis functions
    %----------------------------------------------------------------------
    max_order = 5;
    Ns = 0; % number of basis functions
    for order = 0:max_order
        Ns = Ns + 2 * order + 1;
    end
    A = zeros(N, Ns, 'double');

    %----------------------------------------------------------------------
    % Order = 0 (1)
    %----------------------------------------------------------------------
    if (max_order >= 0)
        A(:,1) = 1;
    end

    %----------------------------------------------------------------------
    % Order = 1 (3)
    %----------------------------------------------------------------------
    if (max_order >= 1)
        A(:,2) = x;
        A(:,3) = y;
        A(:,4) = z;
    end

    %----------------------------------------------------------------------
    % Order = 2 (5)
    %----------------------------------------------------------------------
    if (max_order >= 2)
        A(:,5) = x .* y;
        A(:,6) = z .* y;
        A(:,7) = 2 * z.^2 - (x.^2 + y.^2);
        A(:,8) = x .* z;
        A(:,9) = x.^2 - y.^2;
    end

    %----------------------------------------------------------------------
    % Order = 3 (7)
    %----------------------------------------------------------------------
    if (max_order >= 3)
        A(:,10) = 3 * y .* x.^2 - y.^3;
        A(:,11) = x .* y .* z;
        A(:,12) = 5 * y .* z.^2 - y .* (x.^2 + y.^2 + z.^2);
        A(:,13) = 2 * z.^3 - 3 * z .* (x.^2 + y.^2);
        A(:,14) = 5 * x .* z.^2 - x .* (x.^2 + y.^2 + z.^2);
        A(:,15) = z .* (x.^2 - y.^2);
        A(:,16) = x.^3 - 3 * x .* y.^2;
    end

    %----------------------------------------------------------------------
    % Order = 4 (9)
    %----------------------------------------------------------------------
    if (max_order >= 4)
        r2 = x.^2 + y.^2 + z.^2;
        A(:,17) = x .* y .* (x.^2 - y.^2);
        A(:,18) = z .* y .* (3 * x.^2 - y.^2);
        A(:,19) = (6 * z.^2 - y.^2 - x.^2) .* x .* y;
        A(:,20) = (4 * z.^2 - 3 * y.^2 - 3 * x.^2) .* z .* y;
        A(:,21) = 3 * r2.^2 - 30 * r2 .* z.^2 + 35 * z.^4;
        A(:,22) = (4 * z.^2 - 3 * y.^2 - 3 * x.^2) .* x .* z;
        A(:,23) = (6 * z.^2 - y.^2 - x.^2) .* (x.^2  - y.^2);
        A(:,24) = (x.^2 - 3 * y.^2) .* x .* z;
        A(:,25) = y.^4 - 6 * x.^2 .* y.^2 + x.^4;
    end

    %----------------------------------------------------------------------
    % Order = 5 (11)
    %----------------------------------------------------------------------
    if (max_order >= 5)
        r = sqrt(x.^2 + y.^2 + z.^2);
        A(:,26) = y .* (5 * x.^4 - 10 * x.^2 .* y.^2 + y.^4);
        A(:,27) = 4 * x .* (x - y) .* y .* (x + y) .* z;
        A(:,28) = -y .* (3 * x.^2 - y.^2) .* (r - 3 * z) .* (r + 3 * z);
        A(:,29) = -2 * x .* y .* z .* (r2 - 3 * z.^2);
        A(:,30) = y .* (r2.^2 - 14 * r2 .* z.^2 + 21 * z.^4);
        A(:,31) = 1 / 2 * z .* (15 * r2.^2 - 70 * r2 .* z.^2 + 63 * z.^4);
        A(:,32) = x .* (r2.^2 - 14 * r2 .* z.^2 + 21 * z.^4);
        A(:,33) = -(x - y) .* (x + y) .* z .* (r2 - 3 * z.^2);
        A(:,34) = -x .* (x.^2 - 3 * y.^2) .* (r - 3 * z) .* (r + 3 * z);
        A(:,35) = (x.^2 - 2 * x .* y - y.^2) .* (x.^2 + 2 * x .* y - y.^2) .* z;
        A(:,36) = x .* (x.^4 - 10 * x.^2 .* y.^2 + 5 * y.^4);
    end
    fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));

    %% Smooth a static off-resonance map
    tic; fprintf('(%2d/%2d): Calculating a smooth approximation... ', idx3, N3);
    %----------------------------------------------------------------------
    % Calculate a smooth spherical harmonics approximation to the off-resonance map
    % Given: y = Ax
    % Ordinary least squares (OLS): ||y - Ax||_2^2
    % Weighted least squares (WLS): ||W(y - Ax)||_2^2
    %----------------------------------------------------------------------
    voxel_index = find(mask(:,:,idx3));
    im_echo_ = im_echo(:,:,idx3);
    B0map_raw_ = B0map_raw(:,:,idx3);
    w = abs(im_echo_(voxel_index));
    x_fit = bsxfun(@times, A(voxel_index,:), w) \ (w .* B0map_raw_(voxel_index));
    B0map_fit(:,:,idx3) = reshape(A * x_fit, [N1 N2]);
    fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
end

%% Fill empty space in raw B0 map with spherical harmonics fit
B0map_fill = B0map_fit;
B0map_fill(mask) = B0map_raw(mask);

%% Perform TGV-based denoising on a static off-resonance map
%--------------------------------------------------------------------------
% Primal-dual method for TGV denoising
%--------------------------------------------------------------------------
B0map_tgv = zeros(N1, N2, N3, 'double');
for idx = 1:N3
    tic; fprintf('(%2d/%2d): Performing TGV smoothing (lambda = %g)... ', idx, N3, lambda);
    B0map_tgv(:,:,idx) = TGV_denoise(B0map_fill(:,:,idx), lambda, maxiter_tgv, 0);
    fprintf('done! (%6.4f/%6.4f sec)\n', toc, toc(start_time));
end

%% Fill the holes in the mask
%--------------------------------------------------------------------------
% Fill voids
%--------------------------------------------------------------------------
mask_fill = bwareaopen(mask, 30);
mask_fill = imfill(mask_fill, 'holes');

%--------------------------------------------------------------------------
% Dilate the mask
%--------------------------------------------------------------------------
se = strel('disk', 6);
mask_fill = imdilate(mask_fill, se);

%--------------------------------------------------------------------------
% Fill voids again!
%--------------------------------------------------------------------------
mask_fill = imfill(mask_fill, 'holes');

%% Return a structure containing all intermedicate off-resonance maps
Masks = struct('mask', mask, 'mask_fill', mask_fill);

B0maps = struct('B0map_raw' , B0map_raw , ...
                'B0map_fit' , B0map_fit , ...
                'B0map_fill', B0map_fill, ...
                'B0map_tgv' , B0map_tgv);

%% Display results
if save_figures == 1
    if main_orientation == 0 % sagittal plane
        reorient = @(x) x;
    elseif main_orientation == 1 % coronal plane
        reorient = @(x) x;
    elseif main_orientation == 2 % transverse plane
        reorient = @(x) flip(rot90(x, -1), 2);
    end

    for idx = 1:Ne
        figure('Color', 'w');
        montage(reorient(angle(im_echo(:,:,:,idx) .* mask_fill)*180/pi), 'DisplayRange', []);
        colormap(hsv(256));
        colorbar;
        title(sprintf('Phase of echo %d', idx));
        export_fig(fullfile(output_directory, sprintf('im_echo%d_phase', idx)), '-r400', '-tif');
    end

    figure('Color', 'w');
    montage(reorient(im_max), 'DisplayRange', []);
    export_fig(fullfile(output_directory, 'im_max'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(mask));
    export_fig(fullfile(output_directory, 'mask'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_raw), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Raw static off-resonance map [Hz]')
    export_fig(fullfile(output_directory, 'B0map_raw'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_raw .* mask), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Masked raw static off-resonance map [Hz]')
    export_fig(fullfile(output_directory, 'B0map_mask'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_fit), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Spherical harmonics fit [Hz]')
    export_fig(fullfile(output_directory, 'B0map_fit'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_fit .* mask), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Masked spherical harmonics fit [Hz]')
    export_fig(fullfile(output_directory, 'B0map_fit_mask'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_fill), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Raw + Spherical harmonics fit [Hz]')
    export_fig(fullfile(output_directory, 'B0map_fill'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(mask_fill), 'DisplayRange', []);
    title('Mask without holes')
    export_fig(fullfile(output_directory, 'mask_fill'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_tgv), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('TGV smoothing [Hz]')
    export_fig(fullfile(output_directory, 'B0map_tgv'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_tgv .* mask_fill), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('TGV smoothing x mask [Hz]')
    export_fig(fullfile(output_directory, 'B0map_tgv_mask'), '-r400', '-tif');

    figure('Color', 'w');
    montage(reorient(B0map_fit .* mask_fill), 'DisplayRange', []);
    colormap(hot(256)); colorbar;
    caxis([-60 60]);
    title('Spherical harmonics fit x mask [Hz]')
    export_fig(fullfile(output_directory, 'B0map_fit_mask'), '-r400', '-tif');
end

%end