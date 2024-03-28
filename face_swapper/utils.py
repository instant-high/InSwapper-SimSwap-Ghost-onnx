import cv2
import numpy as np


arcface_template = np.array(
    [
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ]
    ],
    dtype=np.float32,
)


set2_template = np.array(
    [
        [
            [51.6420, 50.1150],
            [57.6170, 49.9900],
            [35.7400, 69.0070],
            [51.1570, 89.0500],
            [57.0250, 89.7020],
        ],
        [
            [45.0310, 50.1180],
            [65.5680, 50.8720],
            [39.6770, 68.1110],
            [45.1770, 86.1900],
            [64.2460, 86.7580],
        ],
        [
            [39.7300, 51.1380],
            [72.2700, 51.1380],
            [56.0000, 68.4930],
            [42.4630, 87.0100],
            [69.5370, 87.0100],
        ],
        [
            [46.8450, 50.8720],
            [67.3820, 50.1180],
            [72.7370, 68.1110],
            [48.1670, 86.7580],
            [67.2360, 86.1900],
        ],
        [
            [54.7960, 49.9900],
            [60.7710, 50.1150],
            [76.6730, 69.0070],
            [55.3880, 89.7020],
            [61.2570, 89.0500],
        ],
    ],
    dtype=np.float32,
)


ffhq_template = np.array(
    [
        [
            [192.98138, 239.94708],
            [318.90277, 240.1936],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118],
        ]
    ],
    dtype=np.float32,
)


template_map = {
    "arcface": (112, arcface_template),
    "set2": (112, set2_template),
    "ffhq": (512, ffhq_template),
}


def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


def get_matrix(lmk, templates):
    if templates.shape[0] == 1:
        return umeyama(lmk, templates[0], True)[0:2, :]
    test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error, best_matrix = float("inf"), []
    for i in np.arange(templates.shape[0]):
        matrix = umeyama(lmk, templates[i], True)[0:2, :]
        error = np.sum(
            np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1))
        )
        if error < min_error:
            min_error, best_matrix = error, matrix
    return best_matrix


def align_crop(img, lmk, image_size, mode="arcface"):
    size, templates = template_map[mode]
    if mode == "arcface" and image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        templates = (templates * ratio) + np.array([8.0 * ratio, 0.0])
    else:
        templates = float(image_size) / size * templates

    matrix = get_matrix(lmk, templates)
    warped = cv2.warpAffine(
        img,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix


def paste_back(img, face, mask, matrix):
    inverse_affine = cv2.invertAffineTransform(matrix)
    h, w = img.shape[0:2]
    face_h, face_w = face.shape[0:2]
    inv_restored = cv2.warpAffine(face, inverse_affine, (w, h))
    inv_restored = inv_restored.astype('float32')
    mask = mask.astype('float32')
    inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
    img = inv_mask * inv_restored + (1 - inv_mask) * img
    return img.clip(0, 255).astype('uint8')