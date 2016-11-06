function ang = geoAngBetween2Vecs(vec1, vec2)
% GEOANGBETWEEN2VECS - Calculate angle between 2 vectors.
%
% Usage:   ang = GEOANGBETWEEN2VECS(vec1, vec2)
%
% Arguments:
%          vec1 - A vector as 3x1 array.
%          vec2 - Another vector as 3x1 array.
%
% Returns:
%          ang - Angle between v1 and v2 in radian.
%
% See Also: SUBSPACE
    if nargin ~= 2
        error('This function has exactly 2 inputs.')
    end
    [M1, N1] = size(vec1);
    [M2, N2] = size(vec2);
    if M1 ~= 3 || N1 ~= 1 || M1 ~= M2 || N1 ~= N2
        error('Size of vec1 and vec2 must be exactly 3x1.')
    end
    ang = subspace(vec1, vec2);
end