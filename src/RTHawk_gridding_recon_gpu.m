function [im_nufft_multislice,header,r_dcs_multislice] = RTHawk_gridding_recon_gpu(data_path, user_opts)
% Written by Nejat Can
% Email: ncan@usc.edu
% Started: 08/21/2023

%% Define constants
gamma = 4257.59 * (1e4 * 2 * pi); % gyromagnetic ratio for 1H [rad/sec/T]

%% Define imaging parameters
%dt  = 2 * 1e-6;   % dwell time [sec]
B0  = 0.55;       % main magnetic field strength [T]
TE  = 2.967 * 1e-3; % echo time [sec]

%% Set reconstruction parameters
tol     = 1e-5; % LSQR tolerance
maxiter = 30;   % maximum number of iterations
Nl      = 19;   % number of spatial basis functions
Lmax    = 30;   % maximum rank of the SVD approximation of a higher-order encoding matrix
L       = 5;    % rank of the SVD approximation of a higher-order encoding matrix
os      = 5;    % oversampling parameter for randomized SVD

%% Read RTHawk .dat file
load(data_path, 'kspace', 'kspace_info', 'raw_dir')
patient_position = 'HFS'; % head first supine

%% Get imaging parameters
% (7654 samples * 16 channels) x (14 interleaves * 5 slices)
Nk = kspace_info.extent(1);           % number of k-space samples
Nc = kspace_info.extent(2);           % number of channels
Ni = kspace_info.kspace.acquisitions; % number of interleaves
Na = user_opts.narm_frame;            % number of arms per frame
M = kspace_info.viewOrder;            % number of spiral arms acquired during scan

%--------------------------------------------------------------------------
% Geometry parameters
%--------------------------------------------------------------------------
FieldOfViewX   = kspace_info.('user_FieldOfViewX'); % [mm]
FieldOfViewY   = kspace_info.('user_FieldOfViewY'); % [mm]
FieldOfViewZ   = kspace_info.('user_FieldOfViewZ'); % [mm]
QuaternionW    = kspace_info.('user_QuaternionW');
QuaternionX    = kspace_info.('user_QuaternionX');
QuaternionY    = kspace_info.('user_QuaternionY');
QuaternionZ    = kspace_info.('user_QuaternionZ');
ResolutionX    = kspace_info.('user_ResolutionX');    % [mm]
ResolutionY    = kspace_info.('user_ResolutionY');    % [mm]
SliceThickness = kspace_info.('user_SliceThickness'); % [mm]
TranslationX   = kspace_info.('user_TranslationX');   % [mm]
TranslationY   = kspace_info.('user_TranslationY');   % [mm]
TranslationZ   = kspace_info.('user_TranslationZ');   % [mm]

%--------------------------------------------------------------------------
% Define reconstruction parameters
%--------------------------------------------------------------------------
%N1 = FieldOfViewX / ResolutionX;
%N2 = FieldOfViewY / ResolutionY;
N1 = user_opts.N1;
N2 = user_opts.N2;
N3 = 1;
N = N1 * N2 * N3;

%--------------------------------------------------------------------------
% Calculate the dwell time [sec]
%--------------------------------------------------------------------------
T = kspace_info.user_readoutTime * 10^-3; % readout duration [sec]
dt = T / Nk;                              % dwell time [sec]

%% Prepare k-space data (Nk x Na x Nc x Nf)
[kspace_echo_1, kspace_echo_2, kx_echo_1, kx_echo_2, ky_echo_1, ...
    ky_echo_2, nframes] = dual_te_split_kspace(kspace, kspace_info, user_opts);

clear kspace_echo_2 kx_echo_2 ky_echo_2 % single echo for now

%--------------------------------------------------------------------------
% Declare Nf parameter
%--------------------------------------------------------------------------
Nf = nframes; % Number of frames in dynamic data

%--------------------------------------------------------------------------
% Permute to achieve (Nk x Na x Nc x Nf)
%--------------------------------------------------------------------------
kspace = permute(kspace_echo_1, [1 2 4 3]); % permute to 

if size(kspace, 1) ~= Nk || size(kspace, 2) ~= Na || size(kspace, 3) ~= Nc || size(kspace, 4) ~= Nf
    error('Your kspace data was reshaped incorrectly.')
end

%% Calculate the maximum k-space value [rad/m]
%--------------------------------------------------------------------------
% Calculate the spatial resolution [m]
%--------------------------------------------------------------------------
spatial_resolution = kspace_info.kspace.spatialResolution(1) * 1e-3;

%--------------------------------------------------------------------------
% Calculate the maximum k-space value [rad/m]
%--------------------------------------------------------------------------
krmax = 2 * pi / spatial_resolution / 2;

%% Get nominal k-space trajectories in the RCS [rad/m] [R,C,S]
%--------------------------------------------------------------------------
% traj: 1 x Nk*Ni*3 x M in [-0.5,0.5]
% [kr(1), kc(1), dcf(1), kr(2), kc(2), dcf(2), ...]
%--------------------------------------------------------------------------

k_rcs_nominal = zeros(Nk, 3, Ni, 'single');
k_rcs_nominal(:,1,:) = reshape(kx_echo_1(:,:, end) * (2 * krmax), [Nk 1 Ni]); % [rad/m]
k_rcs_nominal(:,2,:) = reshape(ky_echo_1(:,:, end) * (2 * krmax), [Nk 1 Ni]); % [rad/m]

%% Calculate a transformation matrix from the RCS to the PCS [R,C,S] <=> [SAG,COR,TRA]
%--------------------------------------------------------------------------
% Convert the quaternion to a rotation matrix representation
% file:///E:/scanning_hawk/doc/RthLibs_doc_html/class_rth_quaternion.html
% Calculate a rotation matrix from RCS to PCS
%--------------------------------------------------------------------------
R_rcs2pcs = zeros(3, 3, 'double');
R_rcs2pcs(1) = 1 - (2 * QuaternionY^2 + 2 * QuaternionZ^2);
R_rcs2pcs(2) = 2 * QuaternionX * QuaternionY + 2 * QuaternionZ * QuaternionW;
R_rcs2pcs(3) = 2 * QuaternionX * QuaternionZ - 2 * QuaternionY * QuaternionW;
R_rcs2pcs(4) = 2 * QuaternionX * QuaternionY - 2 * QuaternionZ * QuaternionW;
R_rcs2pcs(5) = 1 - (2 * QuaternionX^2 + 2 * QuaternionZ^2);
R_rcs2pcs(6) = 2 * QuaternionY * QuaternionZ + 2 * QuaternionX * QuaternionW;
R_rcs2pcs(7) = 2 * QuaternionX * QuaternionZ + 2 * QuaternionY * QuaternionW;
R_rcs2pcs(8) = 2 * QuaternionY * QuaternionZ - 2 * QuaternionX * QuaternionW;
R_rcs2pcs(9) = 1 - (2 * QuaternionX^2 + 2 * QuaternionY^2);

%% Calculate a scaling matrix [m]
scaling_matrix = [ResolutionX 0 0; 0 ResolutionY 0; 0 0 SliceThickness] * 1e-3; % [mm] * [m/1e3mm] => [m]

%% Calculate a rotation matrix from the PCS to the DCS
R_pcs2dcs = siemens_calculate_matrix_pcs_to_dcs(patient_position);

%% Calculate a rotation matrix from the RCS to the DCS
R_rcs2dcs = R_pcs2dcs * R_rcs2pcs;

%% Calculate nominal gradient waveforms in the RCS [mT/m] [R,C,S]
% [rad/m] / [rad/sec/T] / [sec] * [1e3mT/T] => *1e3 [mT/m]
g_rcs_nominal = diff(cat(1, zeros(1,3,Ni), k_rcs_nominal), 1, 1) / (gamma * dt) * 1e3; % [mT/m]

%% Calculate GIRF-predicted gradient waveforms in the RCS [mT/m] [R,C,S]
tRR = 0; % custom clock-shift
sR.R = R_rcs2dcs;
sR.T = 0.55;
[~,g_rcs] = apply_GIRF(permute(g_rcs_nominal, [1 3 2]), dt, sR, tRR); % Nk x Ni x 3
g_rcs = permute(g_rcs, [1 3 2]); % Nk x 3 x Ni

%% Calculate GIRF-predicted gradient waveforms in the DCS [mT/m] [x,y,z]
g_dcs = zeros(Nk, 3, Ni, 'double');
for i = 1:Ni
    g_dcs(:,:,i) = (R_rcs2dcs.' * g_rcs(:,:,i).').'; % Nk x 3
end

%% Calculate GIRF-predicted k-space trajectories in the RCS [rad/m] [R,C,S]
%--------------------------------------------------------------------------
% [rad/sec/T] * [T/1e3mT] * [mT/m] * [sec] => [rad/m]
%--------------------------------------------------------------------------
k_rcs = cumsum(gamma * 1e-3 * g_rcs * dt); % Nk x 3 x Ni [rad/m]

%% Calculate GIRF-predicted k-space trajectories in the DCS [rad/m] [x,y,z]
k_dcs = zeros(Nk, 3, Ni, 'double');
for i = 1:Ni
    k_dcs(:,:,i) = (R_rcs2dcs.' * k_rcs(:,:,i).').'; % Nk x 3
end

%% Calculate a time vector [sec]
t = TE + (0:Nk-1).' * dt; % Nk x 1 [sec]

%% Initialize an NUFFT structure for all interleaves (NUFFT reconstruction)
tstart = tic; fprintf('Initializing structure for NUFFT for all interleaves... ');
% scaled to [-0.5,0.5] and then [-pi,pi]
om = cat(2, reshape(k_rcs(:,1,:), [Nk*Ni 1]), reshape(k_rcs(:,2,:), [Nk*Ni 1])) / (2 * krmax) * (2 * pi); % Nk*Ni x 2
Nd = [N1 N2]; % matrix size
Jd = [6 6];   % kernel size
Kd = Nd * 2;  % oversampled matrix size
nufft_st = nufft_init(om, Nd, Jd, Kd, Nd/2, 'minmax:kb');

%% Calculate a density compensation function (Nk x Ni)
tstart = tic; fprintf('Calculating a density compensation function using sdc3_MAT.c... ');
% [rad/m] / [rad/m] => => [-0.5 0.5]
coords  = permute(k_rcs, [2 1 3]) / (2 * krmax); % Nk x 3 x Ni => 3 x Nk x Ni
numIter = 25; % number of iterations
effMtx  = N1; % the length of one side of the grid matrix
verbose = 0;  % 1:verbose 0:quiet
osf     = 2;  % the grid oversample factor
DCF = sdc3_MAT(coords, numIter, effMtx, verbose, osf);
dcf = DCF / max(DCF(:));

%% Transfer arrays from the CPU to the GPU
%--------------------------------------------------------------------------
% Count only GPU devices that are supported and are currently available
%--------------------------------------------------------------------------
[~,idx] = gpuDeviceCount("available");
gpuDevice(idx(1));

%--------------------------------------------------------------------------
% Transfer arrays from the CPU to the GPU 
%--------------------------------------------------------------------------
dcf_device = gpuArray(dcf);

%% Perform reconstruction per slice
img_nufft   = complex(zeros(N1, N2, Ns, 'double'));
r_dcs = zeros(N, 3, Ns, 'double');
fc1 = zeros(N1, N2, Ns, 'double');

for s = 1:Ns
    %% Get a slice offset in the PCS [m]
    if strcmp(image_ori, 'coronal')
        pcs_offset = [TranslationX; TranslationY + SliceThickness * (s - 1); TranslationZ] * 1e-3; % [mm] * [m/1e3mm] => [m]
    elseif strcmp(image_ori, 'transversal')
        pcs_offset = [TranslationX; TranslationY; TranslationZ + SliceThickness * (s - 1)] * 1e-3; % [mm] * [m/1e3mm] => [m]
    elseif strcmp(image_ori, 'sagittal')
        pcs_offset = [TranslationX - SliceThickness * (s - 1); TranslationY; TranslationZ] * 1e-3; % [mm] * [m/1e3mm] => [m]
    end

    %% Calculate a slice offset in the DCS [m]
    dcs_offset = R_pcs2dcs * pcs_offset; % 3 x 1

    %% Calculate spatial coordinates in the DCS [m]
    %----------------------------------------------------------------------
    % Calculate spatial coordinates in the RCS [m]
    %----------------------------------------------------------------------
    [I1,I2,I3] = ndgrid((1:N1).', (1:N2).', (1:N3).');
    r_rcs = (scaling_matrix * cat(2, I1(:) - (floor(N1/2) + 1), I2(:) - (floor(N2/2) + 1), I3(:) - (floor(N3/2) + 1)).').'; % N x 3

    %----------------------------------------------------------------------
    % Calculate spatial coordinates in the DCS [m]
    %----------------------------------------------------------------------
    r_dcs(:,:,s) = (repmat(dcs_offset, [1 N]) + R_rcs2dcs * r_rcs.').'; % N x 3

    %% Perform NUFFT reconstruction
    cimg_nufft = complex(zeros(N1, N2, Nc, 'double'));
    for c = 1:Nc
        tstart = tic; fprintf('(%2d/%2d) NUFFT reconstruction (c=%2d/%2d)... ', s, Ns, c, Nc);
        cimg_nufft(:,:,c) = nufft_adj(reshape(kspace(:,:,c,s) .* dcf, [Nk*Ni 1]), nufft_st) / sqrt(prod(Nd));
        fprintf('done! (%6.4f sec)\n', toc(tstart));
    end

    %% Estimate coil sensitivity maps
    tstart = tic; fprintf('(%2d/%2d) Estimating coil sensitivity maps with Walsh method... ', s, Ns);
    %----------------------------------------------------------------------
    % IFFT to k-space (k-space <=> image-space)
    %----------------------------------------------------------------------
    kgrid = ifft2c(cimg_nufft);

    %----------------------------------------------------------------------
    % Calculate the calibration region of k-space
    %----------------------------------------------------------------------
    cal_shape = [32 32];
    cal_data = crop(kgrid, [cal_shape Nc]);
    cal_data = bsxfun(@times, cal_data, hamming(cal_shape(1)) * hamming(cal_shape(2)).');

    %----------------------------------------------------------------------
    % Calculate coil sensitivity maps
    %----------------------------------------------------------------------
    cal_im = fft2c(zpad(cal_data, [N1 N2 Nc]));
    csm = ismrm_estimate_csm_walsh(cal_im);

    %% Perform optimal coil combination
    img_nufft(:,:,s) = sum(cimg_nufft .* conj(csm), 3);
end
end