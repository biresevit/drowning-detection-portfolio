from enum import Enum


class ClassesEnum(Enum):
    drowning = 0,
    normal = 1

class CAMERA_TYPE(Enum):
    REOLINK = "reolink"
    CTRONICS = "ctronics"
    DAHUA = "dahua"
    ZHEJIANG = "zheijiang"
    TVT = "tvt"
