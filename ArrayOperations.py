#!/usr/bin/env python
import numpy as np


def mulImgM33(img, mat):
    out = np.zeros((img.shape[0], img.shape[1], 3))
    out[:, :, 0] = img[:, :, 0] * mat[0, 0] + \
        img[:, :, 1] * mat[0, 1] + \
        img[:, :, 2] * mat[0, 2]
    out[:, :, 1] = img[:, :, 0] * mat[1, 0] + \
        img[:, :, 1] * mat[1, 1] + \
        img[:, :, 2] * mat[1, 2]
    out[:, :, 2] = img[:, :, 0] * mat[2, 0] + \
        img[:, :, 1] * mat[2, 1] + \
        img[:, :, 2] * mat[2, 2]
    return out


def mulM33Img(mat, img):
    out = np.zeros((img.shape[0], img.shape[1], 3))
    out[:, :, 0] = mat[0, 0] * img[:, :, 0] + \
        mat[0, 1] * img[:, :, 1] + \
        mat[0, 2] * img[:, :, 2]
    out[:, :, 1] = mat[1, 0] * img[:, :, 0] + \
        mat[1, 1] * img[:, :, 1] + \
        mat[1, 2] * img[:, :, 2]
    out[:, :, 2] = mat[2, 0] * img[:, :, 0] + \
        mat[2, 1] * img[:, :, 1] + \
        mat[2, 2] * img[:, :, 2]
    return out


def mulM34Img(mat, img):
    out = np.zeros((img.shape[0], img.shape[1], 3))
    out[:, :, 0] = mat[0, 0] * img[:, :, 0] + \
        mat[0, 1] * img[:, :, 1] + \
        mat[0, 2] * img[:, :, 2] + \
        mat[0, 3]
    out[:, :, 1] = mat[1, 0] * img[:, :, 0] + \
        mat[1, 1] * img[:, :, 1] + \
        mat[1, 2] * img[:, :, 2] + \
        mat[1, 3]
    out[:, :, 2] = mat[2, 0] * img[:, :, 0] + \
        mat[2, 1] * img[:, :, 1] + \
        mat[2, 2] * img[:, :, 2] + \
        mat[2, 3]
    return out


def dotM33Img(A, B):
    res = np.zeros_like(B)
    C = A.reshape(1, 3, 3)
    res[:, :, 0] = np.sum(C[0, 0, :] * B, axis=2)
    res[:, :, 1] = np.sum(C[0, 1, :] * B, axis=2)
    res[:, :, 2] = np.sum(C[0, 2, :] * B, axis=2)
    return res


def dotImgVec(A, B):
    return (A * B.reshape((1, 1, 3))).sum(axis=2)


def copy(A, B, M):
    shapeA = A.shape
    shapeM = M.shape
    if len(shapeA) == 3 and len(shapeM) == 2:
        h, w = shapeM
        M = M.reshape((h, w, 1))

    MI = np.bitwise_not(M)
    A[...] = M * B + MI * A
