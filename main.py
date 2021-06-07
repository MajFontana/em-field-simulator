import field
import fieldvis
import gui
import fieldcuda




w = gui.Window((1600, 800))
f = fieldcuda.Field((11, 11, 11), 0.01, 0.00000000005)
fv = fieldvis.FieldVisualizer(f)
d = fieldcuda.Dipole(f, 2400000000)
#f.chargedensity[5][5][5] = 1
c = 5




while w.update():
    print("draw")
    w.fill([0, 0, 0])

    w.drawArray(fv.eFieldMagnitude(c, 2, 0.0000000000001), [0.25, 0.5], [0, 0])
    w.drawArray(fv.chargeDensityTransparent(c, 2, 1), [0.25, 0.5], [0, 0])
    w.drawArray(fv.eFieldRgb(c, 2, 0.0000000000001), [0.25, 0.5], [0.25, 0])
    
    w.drawArray(fv.bFieldMagnitude(c, 2, 1), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.currentDensityTransparent(c, 2, 1), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.bFieldRgb(c, 2, 1), [0.25, 0.5], [0.25, 0.5])

    w.drawArray(fv.eFieldMagnitude(c, 0, 0.0000000000001), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.chargeDensityTransparent(c, 0, 1), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.eFieldRgb(c, 0, 0.0000000000001), [0.25, 0.5], [0.75, 0])
    
    w.drawArray(fv.bFieldMagnitude(c, 0, 1), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.currentDensityTransparent(c, 0, 1), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.bFieldRgb(c, 0, 1), [0.25, 0.5], [0.75, 0.5])

    print("update")
    d.update()
    f.update()

    print("wait")
    w.sleep(1)




w.close()
