from src import emfield
from src import fieldvis
from src import gui




w = gui.Window((1600, 800))
f = emfield.Field((21, 21, 21), 0.01, 0.00000000005, "cuda")
fv = fieldvis.FieldVisualizer(f)
d = emfield.Dipole(f, 2400000000)

c = 10




while w.update():
    w.fill([0, 0, 0])

    w.drawArray(fv.eFieldMagnitude(c, 2, 0.00000000000005), [0.25, 0.5], [0, 0])
    w.drawArray(fv.chargeDensityTransparent(c, 2, 1), [0.25, 0.5], [0, 0])
    w.drawArray(fv.eFieldRgb(c, 2, 0.00000000000005), [0.25, 0.5], [0.25, 0])
    
    w.drawArray(fv.bFieldMagnitude(c, 2, 5000), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.currentDensityTransparent(c, 2, 1), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.bFieldRgb(c, 2, 5000), [0.25, 0.5], [0.25, 0.5])

    w.drawArray(fv.eFieldMagnitude(c, 0, 0.00000000000005), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.chargeDensityTransparent(c, 0, 1), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.eFieldRgb(c, 0, 0.00000000000005), [0.25, 0.5], [0.75, 0])
    
    w.drawArray(fv.bFieldMagnitude(c, 0, 5000), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.currentDensityTransparent(c, 0, 1), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.bFieldRgb(c, 0, 5000), [0.25, 0.5], [0.75, 0.5])

    f.update()
    d.update()

    w.sleep(20)




w.close()
