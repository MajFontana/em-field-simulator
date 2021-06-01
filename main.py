import field
import fieldvis
import gui




w = gui.Window((1600, 800))
f = field.Field((21, 21, 21), 0.01, 0.00000000005)
fv = fieldvis.FieldVisualizer(f)
d = field.Dipole(f, 2400000000)




while w.update():
    w.fill([0, 0, 0])

    w.drawArray(fv.eFieldMagnitude(10, 2, 0.00000000000001), [0.25, 0.5], [0, 0])
    w.drawArray(fv.chargeDensityTransparent(10, 2, 1), [0.25, 0.5], [0, 0])
    w.drawArray(fv.eFieldRgb(10, 2, 0.00000000000001), [0.25, 0.5], [0.25, 0])
    
    w.drawArray(fv.bFieldMagnitude(10, 2, 1000), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.currentDensityTransparent(10, 2, 1), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.bFieldRgb(10, 2, 1000), [0.25, 0.5], [0.25, 0.5])

    w.drawArray(fv.eFieldMagnitude(10, 0, 0.00000000000001), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.chargeDensityTransparent(10, 0, 1), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.eFieldRgb(10, 0, 0.00000000000001), [0.25, 0.5], [0.75, 0])
    
    w.drawArray(fv.bFieldMagnitude(10, 0, 1000), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.currentDensityTransparent(10, 0, 1), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.bFieldRgb(10, 0, 1000), [0.25, 0.5], [0.75, 0.5])

    d.update()
    f.update()

    w.sleep(20)




w.close()
