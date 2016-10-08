import glob
from PIL import Image


class GIF(object):
    """
    Class for concatenating images to make GIFs with
    """
    def __init__(self, ima, imb, static=None, processed='images/comp', repeat_end=20):
        self.processed = processed
        self.static = static
        self.rep = repeat_end
        fa = sorted(glob.glob(ima))
        fb = sorted(glob.glob(imb))
        d = len(fb) - len(fa)

        fa = replicate(fa, max([d, 0]) + repeat_end)
        fb = replicate(fb, max([-1 * d, 0]) + repeat_end)
        self.cat(fa, fb)

    def cat(self, fa, fb):
        """
        """
        if self.static:
            statics = [self.static]*len(fa)
            zipped = zip(statics, fa, fb)
        else:
            zipped = zip(fa, fb)

        for count, imgs in enumerate(zipped):
            self.hor_cat(imgs, count)

    def hor_cat(self, imgs, count):
        """
        Combines color images side-by-side.
        """
        images = map(Image.open, imgs)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(self.processed + str(count).zfill(4) + '.png')


def replicate(f, d):
    """
    """
    f.extend([f[-1] for _ in range(d)])
    return f
