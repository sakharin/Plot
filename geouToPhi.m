function phi = geouToPhi(u, W)
% GEOUTOPHI Convert pixel positions u to pan angles phi.
%
% Usage:   u = GEOUTOPHI(u, W)
%
% Arguments:
%          u - An array of pixel position.
%          W   - A width of an equirectangular image.
%
% Returns:
%          phi - An array of pan angles.
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
    phi = u * -2 * pi / W + 2 * pi;
end