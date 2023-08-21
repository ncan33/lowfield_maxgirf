function [im_coil_combined] = nyc_coil_combine_walsh(im_with_coil_dim, dim_of_coil_combination)
    % Function for coil combination. Uses optimal coil combination (Walsh's
    % method)
    % 
    % 'im_with_coil_dim' is the image that needs to be coil combined.
    % 'dim_of_coil_combination' is the dimension which the coil combination
    % needs to take place in.
    %
    % For instance, if the dimensions are X x Y x Nc, one would set
    % 'dim_of_coil_combination' to 3. Another example is, if the dimensions
    % are X x Y x Z x Nc, one would set 'dim_of_coil_combination' to 4.
    
    run server/home/ncan/GitHub/dynamic_maxgirf/dynamic_maxgirf_setup.m
    im_coil_combined = sum(bsxfun(@times, conj(get_sens_map(im_with_coil_dim, '2D')), im_with_coil_dim), dim_of_coil_combination);
end