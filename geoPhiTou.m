function u = geoPhiTou(phi, W)
% GEOPHITOU Convert pan angles phi to pixel positions u.
%
% Usage:   u = GEOPHITOU(phi, W)
%
% Arguments:
%          phi - An array of pan angles.
%          W   - A width of an equirectangular image.
%
% Returns:
%          u - An array of pixel position.
    if nargin < 1
        error('This function needs 1 input.')
    end
    if nargin > 2
        error('This function needs 1 input and 1 optional input.')
    end
    if nargin == 1
        W = 640;
    end
    [WM, WN] = size(W);
    if nargin > 1 && (WM ~= 1 || WN ~= 1 || W <= 0)
        error('W must be a positive number.')
    end
    u = mod((phi - 2 * pi) * W / (-2 * pi), W);
end