function [R, t] = geoEstRtFromE(E, pts1, pts2)
% GEOESTRTFROME Estimate rotation and translation from an essential matrix.
%
% Usage:   [R, t] = GEOESTRTFROME(E, pts1, pts2)
%
% Arguments:
%          E    - An essential matrix such that pts2' * E * pts1 = 0.
%          pts1 - Matched keypoint of camera 1 located at [I | 0] as a 3xN array.
%          pts2 - Matched keypoint of camera 2 located at [R | t] as a 3xN array.
%
% Returns:
%          R - A rotation matrix as a 3x3 array.
%          t - A translation matrix as a 3x1 array.

    % Get 4 solutions
    [U, S, V] = svd(E);
    W = [0, -1, 0; ...
         1,  0, 0; ...
         0,  0, 1];
    Rs = zeros(3, 3, 4);
    Rs(:, :, 1) = det(U * V') * U * W * V.';
    Rs(:, :, 2) = Rs(:, :, 1);
    Rs(:, :, 3) = det(U * V') * U * W.' * V.';
    Rs(:, :, 4) = Rs(:, :, 3);
    ts = zeros(3, 4);
    ts(:, 1) = U(:, 3);
    ts(:, 2) = -ts(:, 1);
    ts(:, 3) =  ts(:, 1);
    ts(:, 4) = -ts(:, 1);

    errs = zeros(1, 4);
    for i = 1:4
        % Solve for translation scale
        s = solveTscale(Rs(:, :, i), ts(:, i), pts1, pts2);

        % Calculate projecttion error
        pts2p = geoProjPts(Rs(:, :, i), abs(s) * ts(:, i), pts1);
        errs(i) = sum(sum((pts2 - pts2p).^2));
    end
    % Choose the solution with lowest error
    sol = find(errs == min(errs));
    R = Rs(:, :, sol(1));
    t = ts(:, sol(1));
end

function s = solveTscale(R, t, pts1, pts2)
    % Linearly solve the scale of translation matrix
    [~, nPts] = size(pts1);
    A = zeros(3 * nPts, 1);
    B = zeros(3 * nPts, 1);
    for iPts = 1:nPts
        Q = pts2(:, iPts) - R * pts1(:, iPts);
        A((iPts - 1) * 3 + 1, 1) = -t(1, 1);
        A((iPts - 1) * 3 + 2, 1) = -t(2, 1);
        A((iPts - 1) * 3 + 3, 1) = -t(3, 1);
        B((iPts - 1) * 3 + 1, 1) = -Q(1, 1);
        B((iPts - 1) * 3 + 2, 1) = -Q(2, 1);
        B((iPts - 1) * 3 + 3, 1) = -Q(3, 1);
    end
    s = linsolve(A, B);
end

