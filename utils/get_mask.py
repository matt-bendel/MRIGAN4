import torch
import numpy as np

from data import transforms


def get_mask(resolution, return_mask=False, R=4, p_m=False, args=None, mask_type=2):
    # total_lines = resolution // R - args.calib_width
    m = np.zeros((resolution, resolution))
    a = None

    if mask_type == 4:
        x = [2, 8]
        y = [32, 16]
        r = np.random.randint(2, 9)
        cw = np.rint(np.interp(r, x, y))
        if cw % 2 != 0:
            cw = cw + 1
        midway = resolution // 2
        s = midway - cw // 2
        e = s + cw
        m[:, s:e] = True
        a = np.random.choice(resolution - cw, resolution // r - cw, replace=False)
        a = np.where(a < s, a, a + args.calib_width)

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
        a = [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
             191, 192, 193, 194, 195,
             196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363]

        # HIGH DIM R=7 GRO
        if R == 7:
            a = [1, 19, 35, 50, 64, 77, 89, 100, 111, 120, 129, 137, 144, 151, 158, 164, 170, 175, 180, 184, 185, 186, 187,
                 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 206, 211, 216, 222, 228, 235, 242, 249,
                 257, 266, 275, 286, 297, 309, 322, 336, 351, 367]

        # HIGH DIM R=6 GRO
        if R == 6:
            a = [1, 15, 28, 41, 52, 63, 74, 84, 93, 102, 110, 118, 125, 132, 138, 144, 150, 156, 161, 166, 171, 175, 180,
                 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 202, 206, 211, 215,
                 220, 225, 230, 236, 242, 248, 254, 261, 268, 276, 284, 293, 302, 312, 323, 334, 345, 358, 371]

        # HIGH DIM R=5 GRO
        if R == 5:
            a = [1, 16, 30, 43, 56, 68, 79, 89, 98, 107, 116, 124, 131, 138, 144, 151, 156, 162, 167, 172, 176, 177, 178,
                 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 214, 219, 224, 230, 235, 242, 248, 255, 262, 270, 279,
                 288, 297, 307, 318, 330, 343, 356, 370]

        # HIGH DIM R=4 GRO
        if R == 4:
            a = [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
                 151, 155, 158, 161, 164,
                 167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
                 223, 226, 229, 233, 236,
                 240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
                 374]

        # HIGH DIM R=3 GRO
        if R == 3:
            a = [1, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 66, 71, 76, 80, 84, 89, 93, 97, 101, 105, 109, 113, 116, 120,
                 124, 127, 130, 134, 137, 140, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 173, 176, 177, 178, 179,
                 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                 201, 202, 203, 204, 205, 206, 207, 208, 210, 213, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 246,
                 249, 252, 256, 259, 262, 266, 270, 273, 277, 281, 285, 289, 293, 297, 302, 306, 310, 315, 320, 324, 329,
                 334, 339, 345, 350, 356, 361, 367, 373, 379]

        # HIGH DIM R=2 GRO
        if R == 2:
            a = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 45, 48, 51, 53, 56, 59, 61, 64, 67, 69, 72, 74,
                 77, 79, 82, 84, 87, 89, 92, 94, 96, 99, 101, 103, 106, 108, 110, 112, 115, 117, 119, 121, 123, 126, 128,
                 130, 132, 134, 136, 138, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171,
                 173, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 213, 215, 217, 219, 221,
                 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 248, 250, 252, 254, 256, 258, 260, 263, 265,
                 267, 269, 271, 274, 276, 278, 280, 283, 285, 287, 290, 292, 294, 297, 299, 302, 304, 307, 309, 312, 314,
                 317, 319, 322, 325, 327, 330, 333, 335, 338, 341, 343, 346, 349, 352, 355, 358, 361, 364, 367, 370, 373,
                 376, 379, 382]

        a = np.array(a)

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
