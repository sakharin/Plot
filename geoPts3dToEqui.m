function ptsEqui = geoPts3dToEqui(pts, W, H)
% GEOPTS3DTOEQUI Convert 3d points to 2d points on an equirectagular image.
%
% Usage:   ptsEqui = GEOPTS3DTOEQUI(pts, W, H)
%
% Arguments:
%          pts - 3xN points.
%          W   - A width of an equirectangular image.
%          H   - A height of an equirectangular image.
%
% Returns:
%          ptsEqui - 2xN point on equirectangular image.
    if nargin < 1
        error('This function needs 1 input.')
    end
    if nargin < 2
        W = 640;
    end
    if nargin < 3
        H = 320;
    end
    if nargin > 3
        error('This function needs 1 input and 2 optional inputs.')
    end
    [ptsM, ptsN] = size(pts);
    if ptsM ~= 3 || ptsN < 1
        error('pts must be a 3xN matrix.')
    end
    [WM, WN] = size(W);
    if nargin > 1 && (WM ~= 1 || WN ~= 1 || W <= 0)
        error('W must be a positive number.')
    end
    [HM, HN] = size(H);
    if nargin > 2 && (HM ~= 1 || HN ~= 1 || H <= 0)
        error('H must be a positive number.')
    end
    [phi, theta] = geoVecToAngs(pts);
    [~, N] = size(pts);
    ptsEqui = zeros(2, N);
    ptsEqui(1, :) = geoPhiTou(phi, W);
    ptsEqui(2, :) = geoThetaTov(theta, H);
end