addpath('tensorlab_2016-03-28')

% rng(42);
close all;
clear; clc;

% load the image
im_size = 16;
path = 'images/dancer.jpg';
img = imread(path);
img = imresize(img, [im_size, im_size]);
img_gray = rgb2gray(img);
x = im2double(img_gray);

n = size(x,1);
m = 2*n - 1;

y = abs(F(zero_pad(x, m))).^2;
z_0 = F_inv(sqrt(y) .* exp(1j * 2 * pi * rand(m)));
% z_0 = zero_pad(x, m); % try to start with the source image

% different options for lambda:
lambda = 2*(rand(1, m^2 - n^2)-0.5) * 1;
% lambda = 0.1 .* randn(1, m^2 - n^2) .* exp(1j * 2 * pi * rand(1, m^2 - n^2));
% lambda = ones(1, m^2 - n^2) * 0.1;
% lambda = ( 2*(rand(1, m^2 - n^2)-0.5) + 1j*2*(rand(1, m^2 - n^2)-0.5) ) * 1;
% lambda = 0.1 * (randn(1, m^2 - n^2) + 1j*randn(1, m^2 - n^2));

f = @(z)G(z, y, n, lambda);
g = @(z)gradG(z, y, n, lambda);

options.Display=10;
options.TolX=-1;
options.TolFun=-1;
options.MaxIter=1e4;

options.LineSearchOptions.alpha=1;        % default: 1
options.LineSearchOptions.c1=1e-4;        % default: 1e-4
options.LineSearchOptions.c2=0.9;         % default: 0.9
options.LineSearchOptions.MaxFunEvals=30; % default: 30

% plot iteration 0
plot_result(x, y, z_0, 0)

% start optimizer
callback_handle = @(z, iteration) callback(x, y, z, iteration);
[z, output] = minf_lbfgs(f,g,z_0,callback_handle,options);

% plot final result
plot_result(x, y,z, output.iterations)

% the callback is called at the end of each iteration
function callback(x, y, z, iteration)
    rho(true); % roll a new rho

    % plot progress
    if mod(iteration, 100) == 0
        plot_result(x,y,z,iteration)
    end
end

function res = rho(roll)
    persistent rho

    % if roll is true, roll a new rho.
    % otherwise return the current rho.
    if isempty(rho) || roll
        rho = rand() * 0.1;
    end

    res = rho;
end

function plot_result(x, y, z, iteration)
    n = size(x,1);
    x_hat = real(z(1:n, 1:n));
    x_diff = abs(x - x_hat);

    f1 = figure(1);
    f1.Position = [50 150 900 500];

    subplot(2,3,1)
    imshow(x)
    title("Original")
    subplot(2,3,4)
    imshow(x_hat)
    title("Recovered")

    subplot(2,3,[2 3])
    imagesc(x_diff)
    colorbar
    pbaspect([1 1 1])
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title("Diff")

    subplot(2,3,[5 6])
    imagesc(abs(z))
    pbaspect([2 2 1])
    colorbar
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title("|Z|")

    % err = norm(z - proj_m(z, y), 'fro');
    % sgtitle(sprintf("Iterations = %d, Err = %1.3e", iteration, err))
    sgtitle(sprintf("Iterations = %d", iteration))

    drawnow;
end

% pad x to size of mxm
function x_pad = zero_pad(x, m)
    n = size(x,1);
    x_pad = zeros(m);
    x_pad(1:n, 1:n) = x;
end

% fourier transform
function f = F(z)
    f = fft2(z);
end

% inverse fourier transform
function f_inv = F_inv(f_z)
    f_inv = ifft2(f_z);
end

% projection operator onto M
function res = proj_m(z,y)
    fft_z = F(z);
    % res = F_inv(sqrt(y) .* (fft_z ./ abs(fft_z)));
    res = F_inv(sqrt(y) .* exp(1j .* angle(fft_z)));
end

% projection operator onto S
function res = proj_s(z,n)
    m = size(z,1);
    res = zeros(m);
    res(1:n,1:n) = z(1:n,1:n);
end

% the operator A
function res = a(z, n)
    m = size(z, 1);
    mask = ones(m, 'logical');
    mask(1:n, 1:n) = false;
    res = z(mask).';
end

% the operator A* (adjoint)
function res = a_star(lambda, n)
    % lambda - vector of length m^2-n^2
    m = sqrt(length(lambda) + n^2);
    res = zeros(m, 'like', lambda);
    idxs = a(reshape(1:m^2, m, m), n);
    res(ind2sub([m,m], idxs)) = conj(lambda);
end

% Frobenius norm squared
function res = norm_F_squared(v)
    v = v(:);
    res = v' * v;
    % res = sum(abs(v).^2, "all");
end

% the objective function
function objective = G(z, y, n, lambda)
    r = rho(false);
    % objective = 0.5 * norm(z - proj_m(z, y), 'fro').^2 + (r/2) * norm(a(z, n) + lambda/r, 2).^2;
    objective = 0.5 * norm_F_squared(z - proj_m(z, y)) + (r/2) * norm_F_squared(a(z, n) + lambda/r);
end

% the gradient of the objective function
function grad = gradG(z, y, n, lambda)
    r = rho(false);
    grad = 0.5 * (z - proj_m(z, y) + r * a_star(a(z, n), n) + a_star(lambda, n));

    % convert from d/dz to d/d(conj(z)) - according to TensorLab docs
    grad = 2 * conj(grad);
end
