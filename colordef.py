
# colorblind-friendily colors

def _c2r(r,g,b,a=1.0):

    return (
        r / 255.0, 
        g / 255.0, 
        b / 255.0, 
        a,
    )

# less saturated colors
CBFC = {
    'r' : _c2r(102,194,165),
    'g' : _c2r(25,141, 98),
    'b' : _c2r(141,160,203),
}

# more saturated colors
CBFC = {
    'r' : _c2r(217,92,2),
    'g' : _c2r(27,158,119),
    'b' : _c2r(117,112,179),
}

# I pick myself
CBFC = {
    'r' : _c2r(255,72,72),
    'g' : "limegreen",#_c2r(95,247,0),
    'b' : _c2r(0,119,255),
}
