import argparse


def int2bool(v):
    try:
        v_int = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("Boolean value must be 0 or 1")
    if v_int not in (0, 1):
        raise argparse.ArgumentTypeError("Boolean value must be 0 or 1")
    return bool(v_int)
