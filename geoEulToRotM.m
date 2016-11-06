function rotm = geoEulToRotM(eul)
% GEOEULTOROTM Calculate Euler rotation matrix XYZ from euler angles.
%
% Usage:   rotm = GEOEULTOROTM(eul)
%
% Arguments:
%          eul - A vector as a 3x1 array of rotation angles in X, Y, and
%                Z direction in radian.
%
% Returns:
%          rotm - 3x3 rotation matrix.
%
% Reference:
%       https://en.wikipedia.org/wiki/Euler_angles
%       https://www.geometrictools.com/Documentation/EulerAngles.pdf
    if nargin ~= 1
        error('This function needs 1 input.')
    end
    [M, N] = size(eul);
    if M ~= 3 || N ~= 1
        error('The input must be a column matrix of size 3x1.')
    end
    c1 = cos(eul(1));
    c2 = cos(eul(2));
    c3 = cos(eul(3));
    s1 = sin(eul(1));
    s2 = sin(eul(2));
    s3 = sin(eul(3));
    rotm = [         c2*c3,         -c2*s3,     s2;...
            c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1;...
            s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2];
end

