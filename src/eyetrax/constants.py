# fmt: off
LEFT_EYE_INDICES = [
    107,  66, 105,  63,  70,
     55,  65,  52,  53,  46,
    468, 469, 470, 471, 472,
    133,  33,
    173, 157, 158, 159, 160, 161, 246,
    155, 154, 153, 145, 144, 163,   7,
    243, 190,  56,  28,  27,  29,  30, 247,
    130,  25, 110,  24,  23,  22,  26, 112,
    244, 189, 221, 222, 223, 224, 225, 113,
    226,  31, 228, 229, 230, 231, 232, 233,
    193, 245, 128, 121, 120, 119, 118, 117,
    111,  35, 124, 143, 156,
]

RIGHT_EYE_INDICES = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
    473, 476, 475, 474, 477,
    362, 263,
    398, 384, 385, 386, 387, 388, 466,
    382, 381, 380, 374, 373, 390, 249,
    463, 414, 286, 258, 257, 259, 260, 467,
    359, 255, 339, 254, 253, 252, 256, 341,
    464, 413, 441, 442, 443, 444, 445, 342,
    446, 261, 448, 449, 450, 451, 452, 453,
    417, 465, 357, 350, 349, 348, 347, 346,
    340, 265, 353, 372, 383,
]

MUTUAL_INDICES = [
      4,   # Nose
     10,   # Very top
    151,   # Forehead
      9,   # Between brow
    152,   # Chin
    234,   # Very left
    454,   # Very right
     58,   # Left jaw
    288,   # Right jaw
]

# Raw-image-space face landmarks for camera-relative position encoding.
# 13 points covering boundary + interior; (x, y) per point = 26 features.
FACE_POSITION_INDICES = [
     10,   # Forehead top
    152,   # Chin
    234,   # Left temple
    454,   # Right temple
     58,   # Left jaw
    288,   # Right jaw
      4,   # Nose tip
      9,   # Between brows
    151,   # Forehead center
    168,   # Nose bridge top
     33,   # Left eye outer corner
    263,   # Right eye outer corner
      1,   # Nose base / upper lip
]
# fmt: on
