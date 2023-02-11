% script for stuffffff
addpath('tensorlab_2016-03-28')

% Load the image
path = 'images/dancer.jpg';
im_size = 50;

img = imread(path);
img = imresize(img, [im_size, im_size]);
img_gray = rgb2gray(img);
x = im2double(img_gray);

n = size(x,1);
m = 2 * n -1;

y = abs(fft2(zero_pad(x, m))) .^ 2;
z_0 = ifft2(sqrt(y) .* exp(1j * 2 * pi * randn(m)));
gamma = ones(1, m^2 - n^2);
r = 1;

f = @(z)G(z,y,n,r, gamma);
g = @(z)grad(z, y, n, r, gamma);

tmp = f(z_0);
tmp_g = g(z_0);

[z,output] = minf_lbfgs(f,g,z_0);
x_hat = abs(z(1:n, 1:n));

imshow(x_hat)



function x_pad = zero_pad(x, m)
    n = size(x,1);
    x_pad = zeros(m);
    x_pad(1:n, 1:n) = x;
end


function res = proj_m(z,y)
    fft_z = fft2(z);
    res = ifft2(sqrt(y) .* fft_z ./ abs(fft_z));
end

function res = a(z, n)
    m = size(z, 1);
    mask = ones(m, 'logical');
    mask(1:n, 1:n) = false;
    res = z(mask).';
end

function res = a_star(gamma, n)
    % gamma - vector of length m^2-n^2
    m = sqrt(length(gamma) + n^2);
    m = cast(m, 'int32');
    res = zeros(m);
    res = cast(res, 'like', gamma);
    idxs = a(reshape(1:m^2, m, m), n);
    res(ind2sub([m,m], idxs)) = conj(gamma);
end

function objective = G(z, y, n, r, gamma)
    proj_m_z = proj_m(z, y);
    z_minus_proj_m_z = z - proj_m_z;
    a_z = a(z, n);
    objective = 0.5 * norm(z_minus_proj_m_z, 'fro')^2 + (r/2) * norm(a_z + gamma)^2; 
    
end


function res = grad(z, y, n, r, gamma)
    proj_m_z = proj_m(z, y);
    z_minus_proj_m_z = z - proj_m_z;
    a_z = a(z, n);
    
    res = 0.5 * (z_minus_proj_m_z + r * a_star(a_z, n) + a_star(gamma, n));
    
    % fix for tensorlab
    res = 2*conj(res);
end
