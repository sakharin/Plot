function eul = geoRotMToEul(rotm)
% GEOROTMTOEUL Calculate Euler angles from euler rotation matrix XYZ.
%
% Usage:   eul = GEOROTMTOEUL(rotm)
%
% Arguments:
%          rotm - 3x3 rotation matrix.
%
% Returns:
%          eul - A vector of rotation angles in X, Y, and Z direction in
%                radian.
%
% Reference:
%       https://en.wikipedia.org/wiki/Euler_angles
%       https://www.geometrictools.com/Documentation/EulerAngles.pdf
    if nargin ~= 1
        error('This function need exactly 1 input.')
    end
    [M, N] = size(rotm);
    if M ~= 3 || N ~= 3
        error('The input must be a 3x3 matrix.')
    end
    r00 = rotm(1, 1);
    r01 = rotm(1, 2);
    r02 = rotm(1, 3);
    r10 = rotm(2, 1);
    r11 = rotm(2, 2);
    r12 = rotm(2, 3);
    %r20 = R(3, 1);
    %r21 = R(3, 2);
    r22 = rotm(3, 3);
    
    if r02 < 1
        if r02 > - 1
            thetaY = asin(r02);
            thetaX = atan2(-r12, r22);
            thetaZ = atan2(-r01, r00);
        else
            thetaY = -pi / 2;
            thetaX = -atan2(r10, r11);
            thetaZ = 0;
        end
    else
        thetaY = pi / 2;
        thetaX = atan2(r10, r11);
        thetaZ = 0;
    end
    eul = [thetaX; thetaY; thetaZ];
end

