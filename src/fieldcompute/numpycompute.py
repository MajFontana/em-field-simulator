import numpy




class FieldEngine:
    def __init__(self, field):
        self.field = field
    
    def compute(self):
        # Compute time derivatives
        efield_dt = (self.field.efield[1] - self.field.efield[0]) * (1 / self.field.timestep)
        bfield_dt = (self.field.bfield[1] - self.field.bfield[0]) * (1 / self.field.timestep)

        for x in range(self.field.size[0]):
            xstart = self.field.modelcenter[0] - x
            xstop = xstart + self.field.size[0]
            for y in range(self.field.size[1]):
                ystart = self.field.modelcenter[1] - y
                ystop = ystart + self.field.size[1]
                for z in range(self.field.size[2]):
                    zstart = self.field.modelcenter[2] - z
                    zstop = zstart + self.field.size[2]
                    
                    # E field due to charge density
                    charge = self.field.chargedensity[x, y, z]
                    if charge != 0:
                        croppedE = self.field.gaussmodel[:, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledE = croppedE * charge
                        self.field.efield[1:] += scaledE

                    """# E field due to B field derivative
                    B_dt = bfield_dt[x, y, z]
                    if any(B_dt):
                        croppedE = self.field.faradaymodel[:, :, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledE = numpy.sum(croppedE * B_dt[:, ...], axis=0)
                        self.field.efield[1:] += scaledE"""

                    # B field due to current density
                    current = self.field.currentdensity[x, y, z]
                    if any(current):
                        croppedB = self.field.amperemodel[:, :, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledB = croppedB * current[:, None, None, None, None, None]
                        self.field.bfield[1:] += numpy.sum(scaledB, 0)

                    """# B field due to E field derivative
                    E_dt = efield_dt[x, y, z]
                    if any(E_dt):
                        croppedB = self.field.maxwellmodel[:, :, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledB = numpy.sum(croppedB * E_dt[:, ...], axis=0)
                        self.field.bfield[1:] += scaledB"""