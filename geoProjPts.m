function projPts = geoProjPts(R, t, pts)
% GEOPROJPTS Project points such that projPts = R * pts + t.
%
% Usage:   projPts = GEOPROJPTS(R, t, pts)
%
% Arguments:
%          R   - 3x3 rotation matrix.
%          t   - 3x1 translation matrix.
%          pts - 3xN points.
%
% Returns:
%          projPts - 3xN projected points.
    if nargin ~= 3
        error('This function needs exactly 3 inputs.')
    end
    [RM, RN] = size(R);
    if RM ~= 3 || RN ~= 3
        error('R must be a 3x3 matrix.')
    end
    [tM, tN] = size(t);
    if tM ~= 3 || tN ~= 1
        error('t must be a 3x1 matrix.')
    end
    [ptsM, ptsN] = size(pts);
    if ptsM ~= 3 || ptsN < 1
        error('pts must be a 3xN matrix.')
    end
    projPts = R * pts;
    projPts(1, :) = projPts(1, :) + t(1);
    projPts(2, :) = projPts(2, :) + t(2);
    projPts(3, :) = projPts(3, :) + t(3);
end