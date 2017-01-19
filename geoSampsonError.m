function err = geoSampsonError(E, x1, x2)
% GEOSAMPSONERROR Calculate Sampson error
%
% Usage : err = GEOSAMPSONERROR(E, x1, x2)
%
% Arguments:
%          E  : A 3x3 essential matrix from frame 1 to frame 2
%          x1 : A 3xN points in frame 1
%          x2 : A 3xN points in frame 2
%
% Returns:
%          err : Sampson error
    x2tEx1 = dot((x2.' * E).', x1);
    Ex1 = E * x1;
    Etx2 = E.' * x2;
    err = sum(x2tEx1.^2 ./ (sum(Ex1(1:2, :).^2) + sum(Etx2(1:2, :).^2)));
end