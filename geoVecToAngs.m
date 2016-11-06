function [phi, theta] = geoVecToAngs(vec)
% GEOVECTOANGS Convert vectors to pan and tilt angles.
%
% Usage:   [phi, theta] = GEOVECTOANGS(vec)
%
% Arguments:
%          vec - A 3xN vectors.
%
% Returns:
%          phi   - 1xN pan angles in radian in range 0 to 2 * pi.
%          theta - 1xN tilt angles in radian in range 0 to pi.
    if nargin ~= 1
        error('This function has exactly 1 input.')
    end
    [M, N] = size(vec);
    if M ~= 3 || N < 1
        error('The size of vec must be 3xN')
    end
    norm = geoNormVec(vec);
    phi = mod(atan2(vec(2, :), vec(1, :)), 2 * pi);
    %phi = atan2(vec(2, :), vec(1, :));
    theta = acos(vec(3, :) ./ norm);
end