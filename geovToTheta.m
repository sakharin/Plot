function theta = geovToTheta(v, H)
% GEOVTOTHETA Convert pixel positions v to tilt angles theta.
%
% Usage:   theta = GEOVTOTHETA(v, H)
%
% Arguments:
%          v - An array of pixel position.
%          H - A height of an equirectangular image.
%
% Returns:
%          theta - An array of tilt angles.
    if nargin < 1
        error('This function needs 1 input.')
    end
    if nargin > 2
        error('This function needs 1 input and 1 optional input.')
    end
    if nargin == 1
        H = 320;
    end
    [HM, HN] = size(H);
    if nargin > 1 && (HM ~= 1 || HN ~= 1 || H <= 0)
        error('H must be a positive number.')
    end
    theta = v * pi / H;
end

