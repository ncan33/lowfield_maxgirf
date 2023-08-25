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
disp(['Nk = ', num2str(Nk)])
disp(['Nk = ', num2str(Ni)])
disp(['Nk = ', num2str(size(kx_echo_1))])
k_rcs_nominal(:,1,:) = reshape(kx_echo_1 * (2 * krmax), [Nk 1 Ni]); % [rad/m]
k_rcs_nominal(:,2,:) = reshape(ky_echo_1 * (2 * krmax), [Nk 1 Ni]); % [rad/m]

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

im_nufft_multislice = 0;
r_dcs_multislice = 0;
header = 0;
return

%{
IN HERE GOES /Volumes/SamsungSSD/Research/FromNam/maxgirf_recon/demo_rthawk_maxgirf_cg_recon.m
IN HERE GOES /Volumes/SamsungSSD/Research/FromNam/maxgirf_recon/demo_rthawk_maxgirf_cg_recon.m
IN HERE GOES /Volumes/SamsungSSD/Research/FromNam/maxgirf_recon/demo_rthawk_maxgirf_cg_recon.m
IN HERE GOES /Volumes/SamsungSSD/Research/FromNam/maxgirf_recon/demo_rthawk_maxgirf_cg_recon.m
IN HERE GOES /Volumes/SamsungSSD/Research/FromNam/maxgirf_recon/demo_rthawk_maxgirf_cg_recon.m
%}

%% Calculate a rotation matrix from PCS to DCS
% KEEP THIS
% KEEP THIS
% KEEP THIS
% KEEP THIS
% KEEP THIS
R_pcs2dcs = siemens_calculate_matrix_pcs_to_dcs(patient_position);

%% Calculate a rotation matrix from GCS to DCS
R_gcs2dcs = R_pcs2dcs * R_gcs2pcs;

%% Calculate nominal gradient waveforms in GCS [G/cm]
%--------------------------------------------------------------------------
% Calculate one spiral interleaf: k in [cycle/cm], g in [G/cm]
%--------------------------------------------------------------------------
Fcoeff = [FOV -FOV * (1 - user_opts.vds_factor / 100)]; % FOV decreases linearly from 30 to 15cm
krmax  = 1 / (2 * (FOV / Nkx));                         % [cycle/cm]
res    = 1 / (2 * krmax) * 10;                          % resolution [mm]
[k_spiral_arm,g_spiral_arm,s_spiral_arm,time] = vdsmex(interleaves, Fcoeff, res, Gmax, Smax, sampling_time, 10000000);
g_spiral_arm = -g_spiral_arm(1:Nk,:); % Nk x 2

%--------------------------------------------------------------------------
% Rotate the spiral interleaf by 360/Ni degrees every TR
% (Re{g_spiral} + 1j * Im{g_spiral}) * (cos(arg) - 1j * sin(arg))
% RO: real =>  Re{g_spiral} * cos(arg) + Im{g_spiral} * sin(arg)
% PE: imag => -Re{g_spiral} * sin(arg) + Im{g_spiral} * cos(arg)
%--------------------------------------------------------------------------
gu_nominal = zeros(Nk, Ni, 'double'); % Nk x Ni [G/cm] PE (gu)
gv_nominal = zeros(Nk, Ni, 'double'); % Nk x Ni [G/cm] RO (gv)
gw_nominal = zeros(Nk, Ni, 'double'); % Nk x Ni [G/cm] SL (gw)
for i = 1:Ni
    arg = 2 * pi / Ni * (i - 1); % [rad]
    gv_nominal(:,i) =  g_spiral_arm(:,1) * cos(arg) + g_spiral_arm(:,2) * sin(arg); % RO (gv)
    gu_nominal(:,i) = -g_spiral_arm(:,1) * sin(arg) + g_spiral_arm(:,2) * cos(arg); % PE (gu)
end

%% Calculate GIRF-predicted gradient waveforms in GCS [G/cm]
g_nominal = cat(3, gu_nominal, gv_nominal);
tRR = 0; % custom clock-shift
sR.R = R_gcs2dcs;
sR.T = header.acquisitionSystemInformation.systemFieldStrength_T;
[~,g_predicted] = apply_GIRF(g_nominal, dt, sR, tRR); % k:[cycle/cm] and g:[G/cm]

%% Change the sign of GIRF-predicted gradient waveforms in GCS [G/cm] [PE,RO,SL]
g_gcs = zeros(Nk, 3, Ni, 'double');
g_gcs(:,1,:) = phase_sign * g_predicted(:,:,1); % [G/cm] PE (gv)
g_gcs(:,2,:) = read_sign  * g_predicted(:,:,2); % [G/cm] RO (gu)
g_gcs(:,3,:) = g_predicted(:,:,3);              % [G/cm] SL (gw)

%% Calculate GIRF-predicted gradient waveforms in DCS [G/cm] [x,y,z]
g_dcs = zeros(Nk, 3, Ni, 'double');
for i = 1:Ni
    g_dcs(:,:,i) = (R_gcs2dcs * g_gcs(:,:,i).').'; % Nk x 3
end

%% Calculate GIRF-predicted k-space trajectories in GCS [rad/m] [PE,RO,SL]
%--------------------------------------------------------------------------
% Numerically integrate the coefficients
% [rad/sec/T] * [G/cm] * [T/1e4G] * [1e2cm/m] * [sec] => [rad/m]
%--------------------------------------------------------------------------
k_gcs = cumsum(gamma * g_gcs * 1e-2 * double(dt)); % [rad/m]

%% Calculate GIRF-predicted k-space trajectories in DCS [rad/m] [x,y,z]
k_dcs = zeros(Nk, 3, Ni, 'double');
for i = 1:Ni
    k_dcs(:,:,i) = (R_gcs2dcs * k_gcs(:,:,i).').'; % Nk x 3
end

%% Calculate GIRF-predicted k-space trajectories in RCS [rad/m] [R,C,S]
k_rcs = zeros(Nk, 3, Ni, 'double');
for i = 1:Ni
    k_rcs(:,:,i) = (R_rcs2gcs.' * k_gcs(:,:,i).').'; % Nk x 3
end

%% Initialize structure for NUFFT for all interleaves (NUFFT reconstruction)
tstart = tic; fprintf('Initializing structure for NUFFT for all interleaves... ');
% scaled to [-0.5,0.5] and then [-pi,pi]
% [rad/m] / ([cycle/cm] * [2pi rad/cycle] * [1e2cm/m]) => [rad/m] / [rad/m] = [unitless] ([-0.5,0.5]) * 2pi => [-pi,pi]
% The definition of FFT is opposite in NUFFT
om = -cat(2, reshape(k_rcs(:,1,:), [Nk*Ni 1]), reshape(k_rcs(:,2,:), [Nk*Ni 1])) / (2 * krmax * 1e2); % Nk*Ni x 2
Nd = [N1 N2]; % matrix size
Jd = [6 6];   % kernel size
Kd = Nd * 2;  % oversampled matrix size
nufft_st = nufft_init(om, Nd, Jd, Kd, Nd/2, 'minmax:kb');
fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

%% Calculate a density compensation function
tstart = tic; fprintf('Calculating a density compensation function based on sdc3_MAT.c... ');
% [rad/m] / ([cycle/cm] * [2pi rad/cycle] * [1e2cm/m]) => [unitless]
coords = permute(k_rcs, [2 1 3]) / (2 * krmax * 2 * pi * 1e2); % Nk x 3 x Ni => 3 x Nk x Ni
coords(3,:,:) = 0;
numIter = 25;
effMtx  = Nkx;
verbose = 0;
DCF = sdc3_MAT(coords, numIter, effMtx, verbose, 2.1);
w = DCF / max(DCF(:));
fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

%% Transfer arrays from the CPU to the GPU
%--------------------------------------------------------------------------
% Count only GPU devices that are supported and are currently available
%--------------------------------------------------------------------------
[~,idx] = gpuDeviceCount("available");
gpuDevice(idx(1));

%--------------------------------------------------------------------------
% Transfer arrays from the CPU to the GPU 
%--------------------------------------------------------------------------
nufft_st_device.n_shift = nufft_st.n_shift;
nufft_st_device.alpha   = {gpuArray(nufft_st.alpha{1}), gpuArray(nufft_st.alpha{2})};
nufft_st_device.beta    = {gpuArray(nufft_st.beta{1}), gpuArray(nufft_st.beta{2})};
nufft_st_device.ktype   = nufft_st.ktype;
nufft_st_device.tol     = nufft_st.tol;
nufft_st_device.Jd      = nufft_st.Jd;
nufft_st_device.Nd      = nufft_st.Nd;
nufft_st_device.Kd      = nufft_st.Kd;
nufft_st_device.M       = nufft_st.M;
nufft_st_device.om      = gpuArray(nufft_st.om);
nufft_st_device.sn      = gpuArray(nufft_st.sn);
nufft_st_device.p       = gpuArray(nufft_st.p);

w_device = gpuArray(w);

%% Calculate the receiver noise matrix
[Psi,inv_L] = calculate_noise_decorrelation_matrix(ismrmrd_noise_fullpath);

%% Perform NUFFT reconstruction per slice
nr_recons = nr_slices * nr_contrasts * nr_phases * nr_repetitions * nr_sets * nr_segments;
im_nufft_multislice = complex(zeros(N1, N2, nr_slices, 'double'));
r_dcs_multislice = zeros(N, 3, nr_slices, 'double');

for idx = 1:nr_recons
    %% Get information about the current slice
    [slice_nr, contrast_nr, phase_nr, repetition_nr, set_nr, segment_nr] = ind2sub([nr_slices nr_contrasts nr_phases nr_repetitions nr_sets nr_segments], idx);
    fprintf('(%2d/%2d): Reconstructing slice (slice = %2d, contrast = %2d, phase = %2d, repetition = %2d, set = %2d, segment = %2d)\n', idx, nr_recons, slice_nr, contrast_nr, phase_nr, repetition_nr, set_nr, segment_nr);

    %% Calculate the actual slice number for Siemens interleaved multislice imaging
    if nr_slices > 1 % multi-slice
        if mod(nr_slices,2) == 0 % even
            offset1 = 0;
            offset2 = 1;
        else % odd
            offset1 = 1;
            offset2 = 0;
        end
        if slice_nr <= ceil(nr_slices / 2)
            actual_slice_nr = 2 * slice_nr - offset1;
        else
            actual_slice_nr = 2 * (slice_nr - ceil(nr_slices/2)) - offset2;
        end
    else
        actual_slice_nr = slice_nr;
    end

    %% Get a list of profiles for all segments
    profile_list = find((raw_data.head.idx.slice      == (slice_nr - 1))      & ...
                        (raw_data.head.idx.contrast   == (contrast_nr - 1))   & ...
                        (raw_data.head.idx.phase      == (phase_nr - 1))      & ...
                        (raw_data.head.idx.repetition == (repetition_nr - 1)) & ...
                        (raw_data.head.idx.set        == (set_nr - 1))        & ...
                        (raw_data.head.idx.segment    == (segment_nr - 1)));

    %% Get a slice offset in PCS from ISMRMRD format
    sag_offset_ismrmrd = double(raw_data.head.position(1,slice_nr)); % [mm]
    cor_offset_ismrmrd = double(raw_data.head.position(2,slice_nr)); % [mm]
    tra_offset_ismrmrd = double(raw_data.head.position(3,slice_nr)); % [mm]

    %% Get a slice offset in PCS from Siemens TWIX format
    if exist(siemens_dat_fullpath, 'file')
        if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}, 'sPosition')
            if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition, 'dSag')
                sag_offset_twix = twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition.dSag; % [mm]
            else
                sag_offset_twix = 0; % [mm]
            end
            if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition, 'dCor')
                cor_offset_twix = twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition.dCor; % [mm]
            else
                cor_offset_twix = 0; % [mm]
            end
            if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition, 'dTra')
                tra_offset_twix = twix.hdr.MeasYaps.sSliceArray.asSlice{actual_slice_nr}.sPosition.dTra; % [mm]
            else
                tra_offset_twix = 0; % [mm]
            end
        else
            sag_offset_twix = 0; % [mm]
            cor_offset_twix = 0; % [mm]
            tra_offset_twix = 0; % [mm]
        end
    end

    %% Use a slice offset in PCS from Siemens TWIX format
    pcs_offset = [sag_offset_twix; cor_offset_twix; tra_offset_twix] * 1e-3; % [mm] * [m/1e3mm] => [m]

    %% Calculate spatial coordinates in DCS [m]
    %----------------------------------------------------------------------
    % Calculate a slice offset in DCS [m]
    %----------------------------------------------------------------------
    dcs_offset = R_pcs2dcs * pcs_offset; % 3 x 1

    %----------------------------------------------------------------------
    % Calculate spatial coordinates in RCS [m]
    %----------------------------------------------------------------------
    [I1,I2,I3] = ndgrid((1:N1).', (1:N2).', (1:N3).');
    r_rcs = (scaling_matrix * cat(2, I1(:) - (floor(N1/2) + 1), I2(:) - (floor(N2/2) + 1), I3(:) - (floor(N3/2) + 1)).').'; % N x 3

    %----------------------------------------------------------------------
    % Calculate spatial coordinates in GCS [m] [PE,RO,SL]
    %----------------------------------------------------------------------
    r_gcs = (R_rcs2gcs * r_rcs.').'; % N x 3

    %----------------------------------------------------------------------
    % Calculate spatial coordinates in DCS [m]
    %----------------------------------------------------------------------
    r_dcs = (repmat(dcs_offset, [1 N]) + R_pcs2dcs * R_gcs2pcs * r_gcs.').'; % N x 3
    r_dcs_multislice(:,:,idx) = r_dcs;

    %% Display slice information
    fprintf('======================= SLICE INFORMATION ========================\n');
    fprintf('slice_nr = %d, actual_slice_nr = %d\n', slice_nr, actual_slice_nr);
    fprintf('phase_sign = %+g, read_sign = %+g\n', phase_sign, read_sign);
    fprintf('---------------------- From Siemens TWIX format ------------------\n');
    fprintf('                   [sag]   %10.5f [mm]\n', sag_offset_twix);
    fprintf('slice offset(PCS): [cor] = %10.5f [mm]\n', cor_offset_twix);
    fprintf('                   [tra]   %10.5f [mm]\n', tra_offset_twix);
    fprintf('---------------------- From ISMRMRD format -----------------------\n');
    fprintf('                   [sag]   %10.5f [mm]\n', sag_offset_ismrmrd);
    fprintf('slice offset(PCS): [cor] = %10.5f [mm]\n', cor_offset_ismrmrd);
    fprintf('                   [tra]   %10.5f [mm]\n', tra_offset_ismrmrd);
    fprintf('---------------------- From Siemens TWIX format ------------------\n');
    fprintf('                   [sag]   [%10.5f %10.5f %10.5f][PE]\n', R_gcs2pcs(1,1), R_gcs2pcs(1,2), R_gcs2pcs(1,3));
    fprintf('R_gcs2pcs        : [cor] = [%10.5f %10.5f %10.5f][RO]\n', R_gcs2pcs(2,1), R_gcs2pcs(2,2), R_gcs2pcs(2,3));
    fprintf('                   [tra]   [%10.5f %10.5f %10.5f][SL]\n', R_gcs2pcs(3,1), R_gcs2pcs(3,2), R_gcs2pcs(3,3));
    fprintf('---------------------- From ISMRMRD format (incorrect!)-----------\n');
    fprintf('                   [sag]   [%10.5f %10.5f %10.5f][PE]\n', R_gcs2pcs_ismrmrd(1,1), R_gcs2pcs_ismrmrd(1,2), R_gcs2pcs_ismrmrd(1,3));
    fprintf('R_gcs2pcs        : [cor] = [%10.5f %10.5f %10.5f][RO]\n', R_gcs2pcs_ismrmrd(2,1), R_gcs2pcs_ismrmrd(2,2), R_gcs2pcs_ismrmrd(2,3));
    fprintf('                   [tra]   [%10.5f %10.5f %10.5f][SL]\n', R_gcs2pcs_ismrmrd(3,1), R_gcs2pcs_ismrmrd(3,2), R_gcs2pcs_ismrmrd(3,3));
    fprintf('------------------------------------------------------------------\n');
    fprintf('                   [ x ]   [%10.5f %10.5f %10.5f][sag]\n', R_pcs2dcs(1,1), R_pcs2dcs(1,2), R_pcs2dcs(1,3));
    fprintf('R_pcs2dcs        : [ y ] = [%10.5f %10.5f %10.5f][cor]\n', R_pcs2dcs(2,1), R_pcs2dcs(2,2), R_pcs2dcs(2,3));
    fprintf('                   [ z ]   [%10.5f %10.5f %10.5f][tra]\n', R_pcs2dcs(3,1), R_pcs2dcs(3,2), R_pcs2dcs(3,3));
    fprintf('------------------------------------------------------------------\n');
    fprintf('                   [ x ]   [%10.5f %10.5f %10.5f][PE]\n', R_gcs2dcs(1,1), R_gcs2dcs(1,2), R_gcs2dcs(1,3));
    fprintf('R_gcs2dcs        : [ y ] = [%10.5f %10.5f %10.5f][RO]\n', R_gcs2dcs(2,1), R_gcs2dcs(2,2), R_gcs2dcs(2,3));
    fprintf('                   [ z ]   [%10.5f %10.5f %10.5f][SL]\n', R_gcs2dcs(3,1), R_gcs2dcs(3,2), R_gcs2dcs(3,3));
    fprintf('==================================================================\n');

    %% Get k-space data (Nk x Ni x Nc)
    %----------------------------------------------------------------------
    % Calculate the index range of readout samples for reconstruction
    %----------------------------------------------------------------------
    index_range = ((discard_pre + 1):(number_of_samples - discard_post)).';

    kspace = complex(zeros(Nk, Ni, Nc, 'double'));
    for idx1 = 1:length(profile_list)
        tstart = tic; fprintf('(%2d/%2d): Reading k-space data (%d/%d)... ', idx, nr_recons, idx1, length(profile_list));
        %------------------------------------------------------------------
        % Determine the interleaf number
        %------------------------------------------------------------------
        interleaf_nr = raw_data.head.idx.kspace_encode_step_1(profile_list(idx1)) + 1;

        %------------------------------------------------------------------
        % Prewhiten k-space data
        %------------------------------------------------------------------
        profile = raw_data.data{profile_list(idx1)}; % number_of_samples x nr_channels
        profile = (inv_L * profile.').';

        %------------------------------------------------------------------
        % Calculate the average of k-space data
        %------------------------------------------------------------------
        profile = profile / nr_averages;

        %------------------------------------------------------------------
        % Accumulate k-space
        %------------------------------------------------------------------
        kspace(:,interleaf_nr,:) = kspace(:,interleaf_nr,:) + reshape(profile(index_range,selected_channels), [Nk 1 Nc]);
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end

    %% Transfer the k-space data array from the CPU to the GPU
    kspace_device = gpuArray(kspace);

    %% Perform NUFFT reconstruction
    imc_nufft_device = complex(zeros(N1, N2, Nc, 'double', 'gpuArray'));
    scale_factor = gpuArray(1 / sqrt(prod(Nd)));
    for c = 1:Nc
        tstart = tic; fprintf('(%2d/%2d): NUFFT reconstruction (c=%2d/%2d)... ', idx, nr_recons, c, Nc);
        imc_nufft_device(:,:,c) = nufft_adj_gpu(reshape(kspace_device(:,:,c) .* w_device, [Nk*Ni 1]), nufft_st_device) * scale_factor;
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end
    imc_nufft = gather(imc_nufft_device);

    %% Calculate coil sensitivity maps
    tstart = tic; fprintf('(%2d/%2d): Calculating coil sensitivity maps with Walsh method... ', idx, nr_recons);
    %----------------------------------------------------------------------
    % IFFT to k-space (k-space <=> image-space)
    %----------------------------------------------------------------------
    kspace_gridded = imc_nufft;
    for dim = 1:2
        kspace_gridded = sqrt(size(kspace_gridded,dim)) * fftshift(ifft(ifftshift(kspace_gridded, dim), [], dim), dim);
    end

    %----------------------------------------------------------------------
    % Calculate the calibration region of k-space
    %----------------------------------------------------------------------
    cal_shape = [32 32];
    cal_data = crop(reshape(kspace_gridded, [N1 N2 Nc]), [cal_shape Nc]);
    cal_data = bsxfun(@times, cal_data, hamming(cal_shape(1)) * hamming(cal_shape(2)).');

    %----------------------------------------------------------------------
    % Calculate coil sensitivity maps
    %----------------------------------------------------------------------
    cal_im = zpad(cal_data, [N1 N2 Nc]);
    for dim = 1:2
        cal_im = 1 / sqrt(size(cal_im,dim)) * fftshift(fft(ifftshift(cal_im, dim), [], dim), dim);
    end
    csm = ismrm_estimate_csm_walsh(cal_im);
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

    %% Perform optimal coil combination
    im_nufft_multislice(:,:,idx) = sum(bsxfun(@times, conj(csm), imc_nufft), 3);
end
end