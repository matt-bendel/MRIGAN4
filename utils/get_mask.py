import torch
import numpy as np

from data import transforms


def get_mask(resolution, return_mask=False, R=4, p_m=False, args=None, mask_type=2):
    # total_lines = resolution // R - args.calib_width
    m = np.zeros((resolution, resolution))
    a = None

    if mask_type == 4:
        R = np.random.randint(2, 8)
        args.calib_width = 32 if R <= 5 else 16
        mask_type = 2

    if mask_type == 3:
        a = np.array(
            [
                28, 34, 40, 41, 55, 72, 80, 84, 92, 107, 152, 164, 177, 184, 185, 186, 187, 188, 189, 190, 191, 192,
                193, 194, 195, 196, 197, 198, 199, 200, 205, 241, 249, 258, 274, 284, 292, 295, 302, 321, 327, 331, 341,
                348, 354, 369, 372, 381
            ]
        )

    # RANDOM MASK GENERATION
    if mask_type == 2:
        midway = resolution // 2
        s = midway - args.calib_width // 2
        e = s + args.calib_width
        m[:, s:e] = True
        a = np.random.choice(resolution - args.calib_width, resolution // R - args.calib_width, replace=False)
        a = np.where(a < s, a, a + args.calib_width)

    # LOW DIM GRO:
    if mask_type == 1:
        # a = np.array(
        #     [
        #         1, 10, 18, 25, 31, 37, 42, 46, 50, 54, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 76, 80, 84, 88,
        #         93, 99, 105, 112, 120
        #     ]
        # )
        # HIGH DIM R=8 GRO
        a = np.array(
            [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
             191, 192, 193, 194, 195,
             196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363])

    # LOW DIM RANDOM R=4 MASK:
    # if mask_type == 3:
    #     a = np.array(
    #         [
    #             0, 2, 4, 10, 24, 25, 36, 39, 49, 50, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 76, 80, 87, 90, 91, 95,
    #             96, 108, 115, 125
    #         ]
    #     )

    # a = np.array(
    #     [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
    #      151, 155, 158, 161, 164,
    #      167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
    #      223, 226, 229, 233, 236,
    #      240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
    #      374])
    # a = np.array(
    #     [1,23,42,60,77,92,105,117,128,138,147,155,162,169,176,182,184,185,186,187,188,189,190,191,192,193,194,195,
    #      196,197,198,199,200,204,210,217,224,231,239,248,258,269,281,294,309,326,344,363])
    m[:, a] = True

    samp = m
    numcoil = 8
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    return mask
