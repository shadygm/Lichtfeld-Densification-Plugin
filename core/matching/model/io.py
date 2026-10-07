from PIL import Image


def check_not_i16(pil_img: Image.Image):
    if pil_img.mode == "I;16":
        raise NotImplementedError("Can't handle 16 bit images")
