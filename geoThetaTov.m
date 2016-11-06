function v= geoThetaTov(theta, H)
% GEOTHETATOV Convert tilt angles theta to pixel positions v.
%
% Usage:   v = GEOTHETATOV(theta, H)
%
% Arguments:
%          theta - An array of tilt angles.
%          H     - A height of an equirectangular image.
%
% Returns:
%          v - An array of pixel position.
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
    v = theta * H / pi;
end