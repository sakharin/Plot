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

        function testEulToRotM(testCase)
            % Solution is not unique beyond 90 degree
            for i = 1:100
                eul = deg2rad(unifrnd(0, 90, 3, 1));
                testCase.verifyEqual(geoRotMToEul(geoEulToRotM(eul)), eul, 'RelTol', 1e-6);
            end
        end

        function testNormVec(testCase)
            % Test with single data
            for i = 1:100
                vec = unifrnd(-10, 10, 3, 1);
                norm = sqrt(vec(1, 1) ^ 2 + vec(2, 1) ^ 2 + vec(3, 1) ^ 2);
                testCase.verifyEqual(geoNormVec(vec), norm, 'RelTol', 1e-6);
            end

            % Test with vector
            vec = unifrnd(-10, 10, 3, 100);
            norm = zeros(1, 100);
            for i = 1:100
                norm(i) = sqrt(vec(1, i) ^ 2 + vec(2, i) ^ 2 + vec(3, i) ^ 2);
            end
            testCase.verifyEqual(geoNormVec(vec), norm, 'RelTol', 1e-6);
        end

        function testNormalizedVec(testCase)
            % Test with single data
            for i = 1:100
                vec = unifrnd(-10, 10, 3, 1);
                norm = sqrt(vec(1, 1) ^ 2 + vec(2, 1) ^ 2 + vec(3, 1) ^ 2);
                normVec = vec / norm;
                testCase.verifyEqual(geoNormalizedVec(vec), normVec, 'RelTol', 1e-6);
            end

            % Test with vector
            vec = unifrnd(-10, 10, 3, 100);
            normVec = zeros(3, 100);
            for i = 1:100
                norm = sqrt(vec(1, i) ^ 2 + vec(2, i) ^ 2 + vec(3, i) ^ 2);
                normVec(:, i) = vec(:, i) / norm; 
            end
            testCase.verifyEqual(geoNormalizedVec(vec), normVec, 'RelTol', 1e-6);
        end

        function testuToPhi(testCase)
            % Test without W
            u = 0:0.1:639;
            testCase.verifyEqual(geoPhiTou(geouToPhi(u)), u, 'RelTol', 1e-6);

            % Test with W
            W = 123;
            u = 0:0.1:W - 1;
            testCase.verifyEqual(geoPhiTou(geouToPhi(u, W), W), u, 'RelTol', 1e-6);
        end

        function testvToTheta(testCase)
            % Test without H
            v = 0:0.1:319;
            testCase.verifyEqual(geoThetaTov(geovToTheta(v)), v, 'RelTol', 1e-6);

            % Test with H
            H = 123;
            v = 0:0.1:H - 1;
            testCase.verifyEqual(geoThetaTov(geovToTheta(v, H), H), v, 'RelTol', 1e-6);
        end

        function testProjPts(testCase)
            % Test with single input
            for i = 1:100
                eul = deg2rad(unifrnd(0, 90, 3, 1));
                R = geoEulToRotM(eul);
                t = unifrnd(-10, 10, 3, 1);
                pts = unifrnd(-10, 10, 3, 1);
                projPts = R * pts + t;
                testCase.verifyEqual(geoProjPts(R, t, pts), projPts, 'RelTol', 1e-6);
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
            testCase.verifyEqual(geoProjPts(R, t, pts), projPts, 'RelTol', 1e-6);
        end

        function testVecCrossToMatrix(testCase)
            vec = [1; 2; 3];
            M = [ 0, -3,  2;
                  3,  0, -1;
                 -2,  1,  0];
            testCase.verifyEqual(geoVecCrossToMatrix(vec), M, 'RelTol', 1e-6);
        end
    end 
end 