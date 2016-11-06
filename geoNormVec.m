function norm = geoNormVec(vec)
% GEONORMVEC Calculate norm of vectors
%
% Usage:   norm = GEONORMVEC(vec)
%
% Arguments:
%          vec - 3xN vectors.
%
% Returns:
%          norm - 1xN norms.
    if nargin ~= 1
        error('This function needs exactly 1 input.')
    end
    [M, N] = size(vec);
    if M ~= 3 || N < 1
        error('The size of vec must be 3xN.')
    end
    norm = sum(vec.^2, 1).^0.5;
end