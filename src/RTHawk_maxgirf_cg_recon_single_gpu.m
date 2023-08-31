function [im_maxgirf_multislice, header, r_dcs_multislice, output] = RTHawk_maxgirf_cg_recon_single_gpu(ismrmrd_noise_path, ismrmrd_data_path, siemens_dat_path, B0map, user_opts)
% Written by Nejat Can
% Email: ncan@usc.edu
% Started 08/22/2023

%% Define constants
gamma = 4257.59 * (1e4 * 2 * pi); % gyromagnetic ratio for 1H [rad/sec/T]

%% Define imaging parameters
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
image_ori = 'coronal';

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

%% Define parameters for reconstruction
%--------------------------------------------------------------------------
% Set the number of samples in the row (r) direction of the RCS
%--------------------------------------------------------------------------
if isfield(user_opts, 'N1')
    N1 = user_opts.N1;
else
    N1 = Nkx;
end

%--------------------------------------------------------------------------
% Set the number of samples in the column (c) direction of the RCS
%--------------------------------------------------------------------------
if isfield(user_opts, 'N2')
    N2 = user_opts.N2;
else
    N2 = Nky;
end 

N3 = Nkz;                % number of samples in the slice (s) direction of the RCS
Ni = nr_phase_encodings; % number of spiral interleaves
N  = N1 * N2;            % total number of voxels in image-space

%--------------------------------------------------------------------------
% Set the number of spatial basis functions
%--------------------------------------------------------------------------
if isfield(user_opts, 'Nl')
    Nl = user_opts.Nl;
else
    Nl = 19;
end

%--------------------------------------------------------------------------
% Calculate the number of samples per spiral arm
%--------------------------------------------------------------------------
if ~isempty(user_opts.discard_pre)
    discard_pre = user_opts.discard_pre;
end
if ~isempty(user_opts.discard_post)
    discard_post = user_opts.discard_post;
end
Nk = number_of_samples - discard_pre - discard_post;

%--------------------------------------------------------------------------
% Select the channels
%--------------------------------------------------------------------------
if isfield(user_opts, 'selected_channels')
    selected_channels = user_opts.selected_channels;
else
    selected_channels = (1:nr_channels).';
end
Nc = length(selected_channels); % number of coils

%--------------------------------------------------------------------------
% Set LSQR parameters
%--------------------------------------------------------------------------
if isfield(user_opts, 'max_iterations')
    max_iterations = user_opts.max_iterations;
else
    max_iterations = 30;
end

if isfield(user_opts, 'tol')
    tol = user_opts.tol;
else
    tol = 1e-5;
end

%--------------------------------------------------------------------------
% Set static_B0_correction
%--------------------------------------------------------------------------
if isfield(user_opts, 'static_B0_correction')
    static_B0_correction = user_opts.static_B0_correction;
else
    static_B0_correction = 1;
end

%--------------------------------------------------------------------------
% Set the maximum rank of the SVD approximation of a higher-order encoding matrix
%--------------------------------------------------------------------------
if isfield(user_opts, 'Lmax')
    Lmax = user_opts.Lmax;
else
    Lmax = 50;
end

%--------------------------------------------------------------------------
% Set the rank of the SVD approximation of a higher-order encoding matrix
%--------------------------------------------------------------------------
if isfield(user_opts, 'L')
    L = user_opts.L;
else
    L = 30;
end

%--------------------------------------------------------------------------
% Set the oversampling parameter for randomized SVD
%--------------------------------------------------------------------------
os = 5; 

%% Prepare a static off-resonance map [Hz]
if isempty(B0map)
    B0map = complex(zeros(N1, N2, nr_slices, 'double'));
end

%% Read a Siemens .dat file
if exist(siemens_dat_path, 'file')
    fprintf('Reading a Siemens .dat file: %s\n', siemens_dat_path);
    twix = mapVBVD(siemens_dat_path);
    if length(twix) > 1
        twix = twix{end};
    end
end

%% Get a slice normal vector from Siemens TWIX format
if exist(siemens_dat_path, 'file')
    %----------------------------------------------------------------------
    % dNormalSag: Sagittal component of a slice normal vector (in PCS)
    %----------------------------------------------------------------------
    if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal, 'dSag')
        dNormalSag = twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal.dSag;
    else
        dNormalSag = 0;
    end

    %----------------------------------------------------------------------
    % dNormalCor: Coronal component of a slice normal vector (in PCS)
    %----------------------------------------------------------------------
    if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal, 'dCor')
        dNormalCor = twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal.dCor;
    else
        dNormalCor = 0;
    end

    %----------------------------------------------------------------------
    % dNormalTra: Transverse component of a slice normal vector (in PCS)
    %----------------------------------------------------------------------
    if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal, 'dTra')
        dNormalTra = twix.hdr.MeasYaps.sSliceArray.asSlice{1}.sNormal.dTra;
    else
        dNormalTra = 0;
    end

    %----------------------------------------------------------------------
    % dRotAngle: Slice rotation angle ("swap Fre/Pha")
    %----------------------------------------------------------------------
    if isfield(twix.hdr.MeasYaps.sSliceArray.asSlice{1}, 'dInPlaneRot')
        dRotAngle = twix.hdr.MeasYaps.sSliceArray.asSlice{1}.dInPlaneRot; % [rad]
    else
        dRotAngle = 0; % [rad]
    end
end

%% Calculate a scaling matrix
scaling_matrix = diag(encoded_resolution) * 1e-3; % [mm] * [m/1e3mm] => [m]

%% Calculate a transformation matrix from RCS to GCS [r,c,s] <=> [PE,RO,SL]
R_rcs2gcs = [0    1    0 ; % [PE]   [0 1 0] * [r]
             1    0    0 ; % [RO] = [1 0 0] * [c]
             0    0    1]; % [SL]   [0 0 1] * [s]

%% Get a rotation matrix from GCS to PCS (ISMRMRD format)
phase_dir = double(raw_data.head.phase_dir(:,1));
read_dir  = double(raw_data.head.read_dir(:,1));
slice_dir = double(raw_data.head.slice_dir(:,1));
R_gcs2pcs_ismrmrd = [phase_dir read_dir slice_dir];

%% Calculate a rotation matrix from GCS to PCS
if exist(siemens_dat_path, 'file')
    [R_gcs2pcs,phase_sign,read_sign] = siemens_calculate_matrix_gcs_to_pcs(dNormalSag, dNormalCor, dNormalTra, dRotAngle);
else
    phase_sign = user_opts.phase_sign;
    read_sign = user_opts.read_sign;
    R_gcs2pcs = [phase_sign * phase_dir read_sign * read_dir slice_dir];
end

%% Calculate a rotation matrix from PCS to DCS
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

%% Calculate the time courses of phase coefficients (Nk x Nl x Ni) [rad/m], [rad/m^2], [rad/m^3]
tstart = tic; fprintf('Calculating the time courses of phase coefficients... ');
k = calculate_concomitant_field_coefficients(reshape(g_dcs(:,1,:), [Nk Ni]), reshape(g_dcs(:,2,:), [Nk Ni]), reshape(g_dcs(:,3,:), [Nk Ni]), Nl, B0, gamma, dt);
fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

%% Calculate a time vector [sec]
t = (0:Nk-1).' * dt; % Nk x 1 [sec]

%% Calculate GIRF-predicted k-space trajectories in GCS [rad/m] [PE,RO,SL]
%--------------------------------------------------------------------------
% Numerically integrate the coefficients
% [rad/sec/T] * [G/cm] * [T/1e4G] * [1e2cm/m] * [sec] => [rad/m]
%--------------------------------------------------------------------------
k_gcs = cumsum(gamma * g_gcs * 1e-2 * dt); % [rad/m]

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

%% Initialize structure for NUFFT per interleaf (MaxGIRF reconstruction)
st = cell(Ni,1);
for i = 1:Ni
    tstart = tic; fprintf('(%2d/%2d): Initializing structure for NUFFT per interleaf... ', i, Ni);
    % scaled to [-0.5,0.5] and then [-pi,pi]
    % [rad/m] / ([cycle/cm] * [2pi rad/cycle] * [1e2cm/m]) => [rad/m] / [rad/m] = [unitless]
    om = -cat(2, k_rcs(:,1,i), k_rcs(:,2,i)) / (2 * krmax * 1e2); % Nk x 2
    Nd = [N1 N2];   % matrix size
    Jd = [6 6];     % kernel size
    Kd = Nd * 2;    % oversampled matrix size
    st{i} = nufft_init(om, Nd, Jd, Kd, Nd/2, 'minmax:kb');
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
end

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
B0map_device = gpuArray(B0map);
k_device = gpuArray(k);
t_device = gpuArray(t);
w_device = gpuArray(w);

st_device = cell(Ni,1);
for i = 1:Ni
    st_device{i}.n_shift = st{i}.n_shift;
    st_device{i}.alpha   = {gpuArray(st{i}.alpha{1}), gpuArray(st{i}.alpha{2})};
    st_device{i}.beta    = {gpuArray(st{i}.beta{1}), gpuArray(st{i}.beta{2})};
    st_device{i}.ktype   = st{i}.ktype;
    st_device{i}.tol     = st{i}.tol;
    st_device{i}.Jd      = st{i}.Jd;
    st_device{i}.Nd      = st{i}.Nd;
    st_device{i}.Kd      = st{i}.Kd;
    st_device{i}.M       = st{i}.M;
    st_device{i}.om      = gpuArray(st{i}.om);
    st_device{i}.sn      = gpuArray(st{i}.sn);
    st_device{i}.p       = gpuArray(st{i}.p);
end

%% Calculate the receiver noise matrix
[Psi,inv_L] = calculate_noise_decorrelation_matrix(ismrmrd_noise_path);

%% Perform NUFFT reconstruction per slice
nr_recons = nr_slices * nr_contrasts * nr_phases * nr_repetitions * nr_sets * nr_segments;
im_maxgirf_multislice = complex(zeros(N1, N2, nr_slices, 'double'));
r_dcs_multislice = zeros(N, 3, nr_slices, 'double');
output = repmat(struct('fc1', [], 's', [], 'flag', [], 'relres', [], 'iter', [], 'resvec', [], 'lsvec', [], 'computation_time_svd', [], 'computation_time_cpr', [], 'computation_time_maxgirf', []), [nr_recons 1]);

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
    if exist(siemens_dat_path, 'file')
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

    %% Calculate concomitant field basis functions (N x Nl) [m], [m^2], [m^3]
    tstart = tic; fprintf('(%2d/%2d): Calculating concomitant field basis functions... ', idx, nr_recons);
    p = calculate_concomitant_field_basis(r_dcs(:,1), r_dcs(:,2), r_dcs(:,3), Nl);
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

    %% Perform NUFFT reconstruction
    imc_nufft = complex(zeros(N1, N2, Nc, 'double'));
    scale_factor = 1 / sqrt(prod(Nd));
    for c = 1:Nc
        tstart = tic; fprintf('(%2d/%2d): NUFFT reconstruction (c=%2d/%2d)... ', idx, nr_recons, c, Nc);
        imc_nufft(:,:,c) = nufft_adj(reshape(double(kspace(:,:,c)) .* w, [Nk*Ni 1]), nufft_st) * scale_factor;
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end

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

    %% Transfer arrays from the CPU to the GPU
    csm_device = gpuArray(csm);
    p_device = gpuArray(p);

    %% Calculate the SVD of the higher-order encoding matrix (Nk x N)
    start_time_svd = tic;
    U_device = complex(zeros(Nk, Lmax, Ni, 'double', 'gpuArray'));
    V_device = complex(zeros(N, Lmax, Ni, 'double', 'gpuArray'));
    s_device = zeros(Lmax, Ni, 'double', 'gpuArray');
    for i = 1:Ni
        tstart = tic; fprintf('(%2d/%2d): Calculating randomized SVD (i=%2d/%2d)... ', idx, nr_recons, i, Ni);
        [U_,S_,V_] = calculate_rsvd_higher_order_encoding_matrix_gpu(k_device(:,4:end,i), p_device(:,4:end), Lmax, os, reshape(B0map_device(:,:,actual_slice_nr), [N 1]), t_device, static_B0_correction);
        U_device(:,:,i) = U_(:,1:Lmax); % U: Nk x Lmax+os => Nk x Lmax
        V_device(:,:,i) = V_(:,1:Lmax) * S_(1:Lmax,1:Lmax)'; % V: N x Lmax+os => N x Lmax
        s_device(:,i) = diag(S_(1:Lmax,1:Lmax));
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end
    computation_time_svd = toc(start_time_svd);

    %% Perform CP-based MaxGIRF reconstruction (conjugate phase reconstruction)
    start_time_cpr = tic;
    b_device = complex(zeros(N, 1, 'double', 'gpuArray'));
    for i = 1:Ni
        tstart = tic; fprintf('(%2d/%2d): Performing CP-based MaxGIRF reconstruction (i=%2d/%2d)... ', idx, nr_recons, i, Ni);

        for c = 1:Nc
            %--------------------------------------------------------------
            % Caclulate d_{i,c}
            %--------------------------------------------------------------
            d = gpuArray(kspace(:,i,c)); % kspace: Nk x Ni x Nc

            %--------------------------------------------------------------
            % Calculate sum_{ell=1}^L ...
            % diag(V(ell,i)) * Fi^H * diag(conj(U(ell,i))) * d_{i,c}
            %--------------------------------------------------------------
            AHd = complex(zeros(N, 1, 'double', 'gpuArray'));
            scale_factor = gpuArray(1 / sqrt(prod(st_device{i}.Nd)));
            for ell = 1:L
                % Preconditioning with density compensation
                FHDuHd = nufft_adj_gpu((conj(U_device(:,ell,i)) .* d) .* w_device(:,i), st_device{i}) * scale_factor;
                AHd = AHd + V_device(:,ell,i) .* reshape(FHDuHd, [N 1]);
            end

            %--------------------------------------------------------------
            % Calculate Sc^H * Ei^H * d_{i,c}
            %--------------------------------------------------------------
            AHd = reshape(conj(csm_device(:,:,c)), [N 1]) .* AHd;

            %--------------------------------------------------------------
            % Calculate b (N x 1)
            %--------------------------------------------------------------
            b_device = b_device + AHd;
        end
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end
    computation_time_cpr = toc(start_time_cpr);

    %% Perform CG-based MaxGIRF reconstruction
    start_time_maxgirf = tic; fprintf('(%2d/%2d): Performing CG-based MaxGIRF reconstruction...\n', idx, nr_recons);
    E = @(x,tr) encoding_lowrank_maxgirf_single_gpu(x, csm_device, U_device(:,1:L,:), V_device(:,1:L,:), w_device, st_device, tr);
    [m_maxgirf_device, flag, relres, iter, resvec] = lsqr(E, b_device, tol, max_iterations, [], [], []); % NL x 1
    im_maxgirf_device = reshape(m_maxgirf_device, [N1 N2]);
    computation_time_maxgirf = toc(start_time_maxgirf);
    fprintf('done! (%6.4f/%6.4f sec)\n', computation_time_maxgirf, toc(start_time));

    %% Collect the output
    im_maxgirf_multislice(:,:,idx,:) = gather(im_maxgirf_device);
    output(idx).fc1    = reshape(k(end,4:end,1) * p(:,4:end).', [N1 N2]) / (2 * pi * T);
    output(idx).s      = gather(s_device);
    output(idx).flag   = flag;
    output(idx).relres = relres;
    output(idx).iter   = iter;
    output(idx).resvec = resvec;
    output(idx).lsvec  = [];
    output(idx).computation_time_svd = computation_time_svd;
    output(idx).computation_time_cpr = computation_time_cpr;
    output(idx).computation_time_maxgirf = computation_time_maxgirf;
end
end