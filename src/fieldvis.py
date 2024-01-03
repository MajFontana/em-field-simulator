import numpy




class FieldVisualizer:
    def __init__(self, field):
        self.field = field
    
    def eFieldRgb(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.efield[1], index, axis)
        col = plane * intensity / 2 + 0.5
        return col
        
    def eFieldMagnitude(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.efield[1], index, axis)
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane)) * intensity
        stitched = numpy.repeat(mag[..., None], 3, 2)
        return stitched

    def bFieldRgb(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.bfield[1], index, axis)
        col = plane * intensity / 2 + 0.5
        return col

    def bFieldMagnitude(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.bfield[1], index, axis)
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane)) * intensity
        stitched = numpy.repeat(mag[..., None], 3, 2)
        return stitched

    def chargeDensity(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.chargedensity, index, axis)
        R = plane * (plane > 0)
        G = numpy.zeros(plane.shape)
        B = numpy.absolute(plane * (plane < 0))
        stitched = numpy.stack([R, G, B], 2) * intensity
        return stitched

    def chargeDensityTransparent(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.chargedensity, index, axis)
        R = plane > 0
        G = numpy.zeros(plane.shape)
        B = plane < 0
        A = numpy.absolute(plane) * intensity
        stitched = numpy.stack([R, G, B, A], 2)
        return stitched

    def currentDensityTransparent(self, index, axis=2, intensity=1):
        plane = numpy.take(self.field.currentdensity, index, axis)
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane))
        RGB = numpy.divide(plane, mag[..., None], where=(mag[..., None] != 0)) / 2 + 0.5
        A = mag * intensity
        stitched = numpy.append(RGB, A[..., None], 2)
        return stitched