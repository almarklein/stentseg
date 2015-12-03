from stentseg.utils import aortamotionpattern
from stentseg.utils.aortamotionpattern import get_motion_pattern, plot_pattern

# Design motion patterns experiment- validation study LSPEAS


# def correctoffset(aa0):
#     offset = aa0[0]
#     aa0[:] = [a - offset for a in aa0]
#     return aa0

def correctoffset(aa0):
    offset = aa0[0]
    A = max(aa0)
    A_no_correction = A-offset
    aa0[:] = [(a - offset)*A/A_no_correction for a in aa0]
    return aa0

def mm2rad(aa0):
    aa0rad = aa0.copy()
    aa0rad[:] = [a/35 for a in aa0] # qrm robot factor 1 rotatie ~ 35 mm
    return aa0rad

## A series

#profile0
tt0,aa0 = get_motion_pattern(A=1.2, T=0.85714, N=20, top=0.35)
aa0cor = aa0.copy()
aa0cor = correctoffset(aa0cor)

#profile1
tt1,aa1 = get_motion_pattern(A=1.2, T=1.2, N=20, top=0.35)
# aa1 = correctoffset(aa1)

#profile2
tt2,aa2 = get_motion_pattern(A=1.2, T=0.6, N=20, top=0.35)
# aa2 = correctoffset(aa2)

#profile3
tt3,aa3 = get_motion_pattern(A=1.2, T=0.8, N=20, top=0.35)
# aa3 = correctoffset(aa3)




#profile4
tt4,aa4 = get_motion_pattern(A=2.0, T=0.85714, N=20, top=0.35)
# aa4 = correctoffset(aa4)

#profile5
tt5,aa5 = get_motion_pattern(A=0.2, T=0.85714, N=20, top=0.35)
# aa5 = correctoffset(aa5)

#profile6
tt6,aa6 = get_motion_pattern(A=1.2, T=0.85714, N=20, top=0.35, extra=(0.4, 0.65, 0.04)) # extra = ampl, t peak top in perc T, sigma in perc T(a * G(t-b)_c)
# aa6 = correctoffset(aa6)

#profile7



import visvis as vv

vv.figure(3)
a = vv.gca()
vv.plot(tt0,aa0, lw=2, lc = 'r')
vv.plot(tt0,aa0cor, lw=2, lc = 'g')
a.axis.showGrid = True

vv.figure(2); vv.clf();

a0 = vv.subplot(241); vv.title('profile0')
plot_pattern(*(tt0,aa0))
a0.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))
a0.axis.showGrid = True

a1 = vv.subplot(242); vv.title('profile1')
plot_pattern(*(tt1,aa1))
a1.axis.showGrid = True
a1.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a2 = vv.subplot(243); vv.title('profile2')
plot_pattern(*(tt2,aa2))
a2.axis.showGrid = True
a2.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a3 = vv.subplot(244); vv.title('profile3')
plot_pattern(*(tt3,aa3))
a3.axis.showGrid = True
a3.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a4 = vv.subplot(245); vv.title('profile4')
plot_pattern(*(tt4,aa4))
a4.axis.showGrid = True
a4.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a5 = vv.subplot(246); vv.title('profile5')
plot_pattern(*(tt5,aa5))
a5.axis.showGrid = True
a5.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a6 = vv.subplot(247); vv.title('profile6')
plot_pattern(*(tt6,aa6))
a6.axis.showGrid = True
a6.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

a7 = vv.subplot(248); vv.title('profile7')
vv.plot([0,0.42857,0.85714], [0,1.2,0])
a7.axis.showGrid = True
a7.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.5,2))

# Get in radians
aa0rad = mm2rad(aa0)
aa1rad = mm2rad(aa1)
aa2rad = mm2rad(aa2)
aa3rad = mm2rad(aa3)
aa4rad = mm2rad(aa4)
aa5rad = mm2rad(aa5)
aa6rad = mm2rad(aa6)






