function pts3D = geoEquiToPt3d(ptsEqui, W, H)
% GEOEQUITOPT3D Convert 2d points on an equirectagular image to 3d points.
%
% Usage:   pts = GEOEQUITOPT3D(ptsEqui, W, H)
%
% Arguments:
%          ptsEqui - 2xN point on equirectangular image.
%          W       - A width of an equirectangular image.
%          H       - A height of an equirectangular image.
%
% Returns:
%          pts - 3xN points.
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
    [WM, WN] = size(W);
    if nargin > 1 && (WM ~= 1 || WN ~= 1 || W <= 0)
        error('W must be a positive number.')
    end
    [HM, HN] = size(H);
    if nargin > 2 && (HM ~= 1 || HN ~= 1 || H <= 0)
        error('H must be a positive number.')
    end
    [ptsEquiM, ptsEquiN] = size(ptsEqui);
    if ptsEquiM ~= 2 || ptsEquiN < 1
        error('ptsEqui must be a 2xN matrix.')
    end
    phi = geouToPhi(ptsEqui(1, :), W);
    theta = geovToTheta(ptsEqui(2, :), H);
    pts3D = geoAngsToVec(phi, theta);
end

