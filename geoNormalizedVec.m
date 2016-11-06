function normVec = geoNormalizedVec(vec)
% GEONORMALIZEDVEC Normalized vectors
%
% Usage:   norm = GEONORMALIZEDVEC(vec)
%
% Arguments:
%          vec - 3xN vectors.
%
% Returns:
%          normVec - 1xN norms.
    if nargin ~= 1
        error('This function needs exactly 1 input.')
    end
    [M, N] = size(vec);
    if M ~= 3 || N < 1
        error('The size of vec must be 3xN.')
    end
    norm = geoNormVec(vec);
    normVec = zeros(3, N);
    normVec(1, :) = vec(1, :) ./ norm;
    normVec(2, :) = vec(2, :) ./ norm;
    normVec(3, :) = vec(3, :) ./ norm;
end