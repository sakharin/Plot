function vec = geoAngsToVec(phi, theta)
% GEOANGSTOVEC Calculate unit vectors from angles.
%
% Usage:   vec = GEOANGSTOVEC(phi, theta)
%
% Arguments:
%          phi   - 1xN pan angles in radian in range 0 to 2 * pi.
%          theta - 1xN tilt angles in radian in range 0 to pi.
%
% Returns:
%          vec - 3xN unit vectors.
    if nargin ~= 2
        error('This function has exactly 2 inputs.')
    end
    [M1, N1] = size(phi);
    [M2, N2] = size(theta);
    if M1 ~= 1 || M1 ~= M2 || N1 ~= N2
        error('Size of phi and theta must be equal.')
    end
    vec = zeros(3, N1); 
    vec(1, :) = sin(theta) .* cos(phi);
    vec(2, :) = sin(theta) .* sin(phi);
    vec(3, :) = cos(theta);
end