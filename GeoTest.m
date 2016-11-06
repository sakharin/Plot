% Test case for geo functions
% Execute by :
%     testCase = GeoTest; res = run(testCase)
classdef GeoTest < matlab.unittest.TestCase
    % Test cases for Geo functions.

    methods (Test)
        function testAngBetween2Vecs(testCase)
            % Angle between x- and y-axis is pi / 2.
            vec1 = [1; 0; 0];
            vec2 = [0; 1; 0];
            testCase.verifyEqual(geoAngBetween2Vecs(vec1, vec2), pi / 2);
        end

        function testAngsToVec(testCase)
            % Test with single data
            phis   = [ 0.0,  0.0,  0.0,  0.5,  1.0,  1.5,  2.0] * pi;
            thetas = [ 0.0,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5] * pi;
            vecs   = [ 0.0,  1.0,  0.0,  0.0, -1.0,  0.0,  1.0; ...
                       0.0,  0.0,  0.0,  1.0,  0.0, -1.0,  0.0; ...
                       1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0];
            [~, N] = size(phis);
            for i = 1:N
                phi   = phis(1, i);
                theta = thetas(1, i);
                vec   = vecs(:, i);
                testCase.verifyEqual(geoAngsToVec(phi, theta), vec, 'AbsTol', 1e-6);
            end

            % Test with vector
            testCase.verifyEqual(geoAngsToVec(phis, thetas), vecs, 'AbsTol', 1e-6);
        end

        function testVecToAngs(testCase)
            % Test with single data
            phis   = [ 0.0,  0.0,  0.0,  0.5,  1.0,  1.5,  2.0] * pi;
            thetas = [ 0.0,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5] * pi;
            vecs   = [ 0.0,  1.0,  0.0,  0.0, -1.0,  0.0,  1.0; ...
                       0.0,  0.0,  0.0,  1.0,  0.0, -1.0,  0.0; ...
                       1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0];
            dists  = [ 1.2,  1.5,  9.2,  0.4,  5.6,  8.3,  6.7];
            vecs = repmat(dists, 3, 1) .* vecs; % Make non-unit vectors
            [~, N] = size(phis);
            for i = 1:N
                phi   = phis(1, i);
                theta = thetas(1, i);
                vec   = vecs(:, i);
                [resPhi, resTheta] = geoVecToAngs(vec);
                testCase.verifyEqual([mod(phi, 2 * pi), theta], [resPhi, resTheta], 'AbsTol', 1e-6);
            end

            % Test with vector
            [resPhis, resThetas] = geoVecToAngs(vecs);
            testCase.verifyEqual([mod(phis, 2 * pi), thetas], ...
                                 [resPhis, resThetas], 'AbsTol', 1e-6);
        end

        function testEquiToPt3d(testCase)
            % Test with single data
            for i = 1:100
                H = unifrnd(1, 1000);
                W = H * 2;
                equi = zeros(2, 1);
                equi(1, 1) = unifrnd(0, W - 1);
                equi(2, 1) = unifrnd(0, H - 1);
                p = geouToPhi(equi(1, 1), W);
                t = geovToTheta(equi(2, 1), H);
                vec = geoAngsToVec(p, t);
                testCase.verifyEqual(geoEquiToPt3d(equi, W, H), vec, 'AbsTol', 1e-6);
            end

            % Test with vector
            for i = 1:100
                H = unifrnd(1, 1000);
                W = H * 2;
                equi = zeros(2, 100);
                equi(1, :) = unifrnd(0, W - 1, 1, 100);
                equi(2, :) = unifrnd(0, H - 1, 1, 100);
                p = geouToPhi(equi(1, :), W);
                t = geovToTheta(equi(2, :), H);
                vec = geoAngsToVec(p, t);
                testCase.verifyEqual(geoEquiToPt3d(equi, W, H), vec, 'AbsTol', 1e-6);
            end
        end

        function testPt3dToEqui(testCase)
            % Test with single data
            for i = 1:100
                H = unifrnd(1, 1000);
                W = H * 2;
                vec = unifrnd(-100, 100, 3, 1);
                [p, t] = geoVecToAngs(vec);
                equi = zeros(2, 1);
                equi(1, 1) = geoPhiTou(p, W);
                equi(2, 1) = geoThetaTov(t, H);
                testCase.verifyEqual(geoPts3dToEqui(vec, W, H), equi, 'AbsTol', 1e-6);
            end

            % Test with vector
            for i = 1:100
                H = unifrnd(1, 1000);
                W = H * 2;
                vecs = unifrnd(-100, 100, 3, 100);
                [p, t] = geoVecToAngs(vecs);
                equis = zeros(2, 100);
                equis(1, :) = geoPhiTou(p, W);
                equis(2, :) = geoThetaTov(t, H);
                testCase.verifyEqual(geoPts3dToEqui(vecs, W, H), equis, 'AbsTol', 1e-6);
            end
        end

        function testEulToRotM(testCase)
            % Solution is not unique beyond 90 degree
            for i = 1:100
                eul = deg2rad(unifrnd(0, 90, 3, 1));
                testCase.verifyEqual(geoRotMToEul(geoEulToRotM(eul)), eul, 'AbsTol', 1e-6);
            end
        end

        function testNormVec(testCase)
            % Test with single data
            for i = 1:100
                vec = unifrnd(-10, 10, 3, 1);
                norm = sqrt(vec(1, 1) ^ 2 + vec(2, 1) ^ 2 + vec(3, 1) ^ 2);
                testCase.verifyEqual(geoNormVec(vec), norm, 'AbsTol', 1e-6);
            end

            % Test with vector
            vec = unifrnd(-10, 10, 3, 100);
            norm = zeros(1, 100);
            for i = 1:100
                norm(i) = sqrt(vec(1, i) ^ 2 + vec(2, i) ^ 2 + vec(3, i) ^ 2);
            end
            testCase.verifyEqual(geoNormVec(vec), norm, 'AbsTol', 1e-6);
        end

        function testNormalizedVec(testCase)
            % Test with single data
            for i = 1:100
                vec = unifrnd(-10, 10, 3, 1);
                norm = sqrt(vec(1, 1) ^ 2 + vec(2, 1) ^ 2 + vec(3, 1) ^ 2);
                normVec = vec / norm;
                testCase.verifyEqual(geoNormalizedVec(vec), normVec, 'AbsTol', 1e-6);
            end

            % Test with vector
            vec = unifrnd(-10, 10, 3, 100);
            normVec = zeros(3, 100);
            for i = 1:100
                norm = sqrt(vec(1, i) ^ 2 + vec(2, i) ^ 2 + vec(3, i) ^ 2);
                normVec(:, i) = vec(:, i) / norm;
            end
            testCase.verifyEqual(geoNormalizedVec(vec), normVec, 'AbsTol', 1e-6);
        end

        function testuToPhi(testCase)
            % Test without W
            u = 0:0.1:639;
            testCase.verifyEqual(geoPhiTou(geouToPhi(u)), u, 'AbsTol', 1e-6);

            % Test with W
            W = 123;
            u = 0:0.1:W - 1;
            testCase.verifyEqual(geoPhiTou(geouToPhi(u, W), W), u, 'AbsTol', 1e-6);
        end

        function testvToTheta(testCase)
            % Test without H
            v = 0:0.1:319;
            testCase.verifyEqual(geoThetaTov(geovToTheta(v)), v, 'AbsTol', 1e-6);

            % Test with H
            H = 123;
            v = 0:0.1:H - 1;
            testCase.verifyEqual(geoThetaTov(geovToTheta(v, H), H), v, 'AbsTol', 1e-6);
        end

        function testProjPts(testCase)
            % Test with single input
            for i = 1:100
                eul = deg2rad(unifrnd(0, 90, 3, 1));
                R = geoEulToRotM(eul);
                t = unifrnd(-10, 10, 3, 1);
                pts = unifrnd(-10, 10, 3, 1);
                projPts = R * pts + t;
                testCase.verifyEqual(geoProjPts(R, t, pts), projPts, 'AbsTol', 1e-6);
            end

            % Test with vectors
            eul = deg2rad(unifrnd(0, 90, 3, 1));
            R = geoEulToRotM(eul);
            t = unifrnd(-10, 10, 3, 1);
            pts = unifrnd(-10, 10, 3, 100);
            projPts = zeros(3, 100);
            for i = 1:100
                projPts(:, i) = R * pts(:, i) + t;
            end
            testCase.verifyEqual(geoProjPts(R, t, pts), projPts, 'AbsTol', 1e-6);
        end

        function testVecCrossToMatrix(testCase)
            vec = [1; 2; 3];
            M = [ 0, -3,  2;
                  3,  0, -1;
                 -2,  1,  0];
            testCase.verifyEqual(geoVecCrossToMatrix(vec), M, 'AbsTol', 1e-6);
        end
    end
end
