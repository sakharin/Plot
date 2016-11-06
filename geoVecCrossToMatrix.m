function rotm = geoVecCrossToMatrix(vec)
% GEOVECCROSSTOMATRIX Calculate Euler rotation matrix XYZ from euler angles.
%
% Usage:   rotm = GEOVECCROSSTOMATRIX(vec)
%
% Arguments:
%          vec - A vector.
%
% Returns:
%          rotm - 3x3 cross product matrix.
    if nargin ~= 1
        error('This function has exactly 1 inputs.')
    end
    [M, N] = size(vec);
    if M ~= 3 || N ~= 1
        error('Size of vec must be exactly 3x1.')
    end
    a1 = vec(1);
    a2 = vec(2);
    a3 = vec(3);
    rotm = [  0, -a3,  a2;...
             a3,   0, -a1;...
            -a2,  a1,   0];
end