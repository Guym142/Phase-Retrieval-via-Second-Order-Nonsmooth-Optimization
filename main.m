% script for stuffffff
addpath('tensorlab_2016-03-28')

% Load the image
path = 'images/dancer.jpg';
im_size = 100;

img = imread(path);
img = imresize(img, [im_size, im_size]);
img_gray = rgb2gray(img);
x = im2double(img_gray);

n = size(x,1);
m = (2 * n);

y = abs(fftshift(fft2(zero_pad(x, m))) ./ m) .^ 2;
z_0 = ifftshift(ifft2(sqrt(y) .* exp(1j * 2 * pi * randn(m)))) .* m;


gamma = ones(1, m^2 - n^2);
scale =  0.1;
global r
r = rand() * scale;

f = @(z)G(z,y,n, gamma);
g = @(z)grad(z, y, n, gamma);

tmp = f(z_0);
tmp_g = g(z_0);

[z,output] = minf_lbfgs(f,g,z_0);
f_final = f(z);
x_hat = abs(z(1:n, 1:n));

imshow(x_hat)



function x_pad = zero_pad(x, m)
    n = size(x,1);
    x_pad = zeros(m);
    x_pad(1:n, 1:n) = x;
end


function res = proj_m(z,y)
    m = size(y,1);
    fft_z = fftshift(fft2(z)) ./ m;
    res = ifftshift(ifft2(sqrt(y) .* fft_z ./ abs(fft_z))) .* m;
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

function objective = G(z, y, n, gamma)
    global r
%     r = rand() * 0.1;
    proj_m_z = proj_m(z, y);
    z_minus_proj_m_z = z - proj_m_z;
    a_z = a(z, n);
    objective = 0.5 * norm(z_minus_proj_m_z, 'fro')^2 + (r/2) * norm(a_z + gamma)^2;   
end


function res = grad(z, y, n, gamma)
    global r
    disp(r)
    proj_m_z = proj_m(z, y);
    z_minus_proj_m_z = z - proj_m_z;
    a_z = a(z, n);
    
    res = 0.5 * (z_minus_proj_m_z + r * a_star(a_z, n) + a_star(gamma, n));
    
    % fix for tensorlab
    res = 2*conj(res);
end
