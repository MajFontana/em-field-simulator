from src import emfield
from src import fieldvis
from src import gui




w = gui.Window((1600, 800))
f = emfield.Field((71, 71, 1), 0.005, 0.00000000001, "numpy")
fv = fieldvis.FieldVisualizer(f)
d = emfield.Dipole(f, 2400000000, 1e-16)

c1 = 0
c2 = 35




intensity = 1e16
e_field_intensity = 6e2
b_field_intensity = 3e19

while w.update():
    w.fill([0, 0, 0])

    w.drawArray(fv.eFieldMagnitude(c1, 2, e_field_intensity), [0.25, 0.5], [0, 0])
    w.drawArray(fv.chargeDensityTransparent(c1, 2, intensity), [0.25, 0.5], [0, 0])
    w.drawArray(fv.eFieldRgb(c1, 2, e_field_intensity), [0.25, 0.5], [0.25, 0])
    
    w.drawArray(fv.bFieldMagnitude(c1, 2, b_field_intensity), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.currentDensityTransparent(c1, 2, intensity), [0.25, 0.5], [0, 0.5])
    w.drawArray(fv.bFieldRgb(c1, 2, b_field_intensity), [0.25, 0.5], [0.25, 0.5])

    w.drawArray(fv.eFieldMagnitude(c2, 0, e_field_intensity), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.chargeDensityTransparent(c2, 0, intensity), [0.25, 0.5], [0.5, 0])
    w.drawArray(fv.eFieldRgb(c2, 0, e_field_intensity), [0.25, 0.5], [0.75, 0])
    
    w.drawArray(fv.bFieldMagnitude(c2, 0, b_field_intensity), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.currentDensityTransparent(c2, 0, intensity), [0.25, 0.5], [0.5, 0.5])
    w.drawArray(fv.bFieldRgb(c2, 0, b_field_intensity), [0.25, 0.5], [0.75, 0.5])

    f.update()
    d.update()

    w.sleep(20)




w.close()
