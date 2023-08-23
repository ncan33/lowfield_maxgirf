function [kspace_echo_1, kspace_echo_2, kx_echo_1, kx_echo_2, ky_echo_1, ...
    ky_echo_2, nframes] = dual_te_split_kspace(kspace, kspace_info, user_opts)
    % splits dual TE kspace data
    Nk = kspace_info.extent(1);           % number of k-space samples
    Nc = kspace_info.extent(2);           % number of channels
    Ni = kspace_info.kspace.acquisitions; % number of interleaves
    narm_frame = user_opts.narm_frame;    % number of arms per frame
    
    view_order = kspace_info.viewOrder; % view order

    echo_idx = view_order > Ni;
    echo_idx = echo_idx + 1; % if the element in echo_idx == 1, then echo_1,
                             % and when echo_idx == 2, then echo_2
    %--------------------------------------------------------------------------
    % Split kspace
    %--------------------------------------------------------------------------
    kspace_echo_1 = kspace(:, echo_idx == 1, :); % raw kspace data for echo 1
    kspace_echo_2 = kspace(:, echo_idx == 2, :); % raw kspace data for echo 2
    clear kspace

    narm_total = min(size(kspace_echo_1, 2), size(kspace_echo_2, 2)); % narm after splitting kspace

    %--------------------------------------------------------------------------
    % Orgnize the data to frames
    %--------------------------------------------------------------------------
    nframes = floor(narm_total / narm_frame);
    narm_total = nframes * narm_frame;

    kspace_echo_1(:, narm_total + 1 : end, :) = []; % discard excess kspace
    kspace_echo_2(:, narm_total + 1 : end, :) = []; % discard excess kspace

    kspace_echo_1  = reshape(kspace_echo_1, [Nk, narm_frame, nframes, Nc]);
    kspace_echo_2  = reshape(kspace_echo_2, [Nk, narm_frame, nframes, Nc]);
    
    %--------------------------------------------------------------------------
    % k-space trajectory and view order
    %--------------------------------------------------------------------------
    view_order_echo_1 = view_order(echo_idx == 1); % view_order for echo_1
    view_order_echo_2 = view_order(echo_idx == 2); % view_order for echo_2
    view_order_echo_1(narm_total + 1 : end) = []; % discard excess views
    view_order_echo_2(narm_total + 1 : end) = []; % discard excess views
    view_order_echo_1  = reshape(view_order_echo_1, [narm_frame, nframes]);
    view_order_echo_2  = reshape(view_order_echo_2, [narm_frame, nframes]);

    kx = kspace_info.kx_GIRF; % kx
    ky = kspace_info.ky_GIRF; % ky

    kx_echo_1 = zeros(Nk, narm_frame, nframes); %kx_echo_1 = zeros(size(kspace_echo_1,1), size(kspace_echo_1,2), size(kspace_echo_1,3));
    kx_echo_2 = kx_echo_1;
    ky_echo_1 = kx_echo_1;
    ky_echo_2 = kx_echo_1;

    for i = 1:narm_frame
        for j = 1:nframes
            kx_echo_1(:,i,j) = kx(:, view_order_echo_1(i,j));
            kx_echo_2(:,i,j) = kx(:, view_order_echo_2(i,j) - 10);

            ky_echo_1(:,i,j) = ky(:, view_order_echo_1(i,j));
            ky_echo_2(:,i,j) = ky(:, view_order_echo_2(i,j) - 10);
        end
    end
end
