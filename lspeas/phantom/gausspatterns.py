""" Design motion patterns for CT experiments QRM robot

"""

from stentseg.utils import aortamotionpattern
from stentseg.utils.aortamotionpattern import get_motion_pattern, plot_pattern


def correctoffset(aa0):
    """set t=0 to zero and scale signal
    """
    offset = aa0[0]
    A = max(aa0)
    A_no_correction = A-offset
    aa0[:] = [(a - offset)*A/A_no_correction for a in aa0]
    return

def mm2rad(aa0):
    aa0rad = aa0.copy()
    aa0rad[:] = [a/35 for a in aa0] # qrm robot factor 1 rotatie ~ 35 mm
    return aa0rad

def patternToExcel(tt0, aa0, profile='profile0'):
    """Create file and add a worksheet or overwrite existing
    """
    import xlsxwriter
    # https://pypi.python.org/pypi/XlsxWriter
#     workbook = xlsxwriter.Workbook(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Phantom validation\QRM Software\LSPEAS-motion-profiles\patternoutput.xlsx')
    workbook = xlsxwriter.Workbook(r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Phantom validation\QRM Software\LSPEAS-motion-profiles\patternoutput.xlsx')
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('C:E', 15)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    # write title
    worksheet.write('A1', profile, bold)
    worksheet.write('C1', 'tt (tijd in s)', bold)
    worksheet.write('D1', 'aarad (pos in rad)', bold)
    worksheet.write('E1', 'aa (pos in mm)', bold)
    # get amplitudes in radians
    aa0rad = mm2rad(aa0)
    # add last point t = T
    T = tt0[-1] + tt0[1]
    tt0.append(T)
    aa0.append(0)
    aa0rad.append(0)
    # write 'storeOutput'
    rowstarttt, rowstartaarad, rowstartaa  = 2, 2, 2
    for t in tt0:
        worksheet.write_row(rowstarttt, 2, [t]) # row, columnm, variable
        rowstarttt += 1
    for arad in aa0rad:
        worksheet.write_row(rowstartaarad, 3, [arad]) # row, columnm, variable
        rowstartaarad += 1
    for a in aa0:
        worksheet.write_row(rowstartaa, 4, [a]) # row, columnm, variable
        rowstartaa += 1
        
    workbook.close()
    return

    
## A series

#profile0
tt0,aa0 = get_motion_pattern(A=1.2, T=0.85714, N=20, top=0.35)
correctoffset(aa0)

#profile1
tt1,aa1 = get_motion_pattern(A=1.0, T=1.2, N=20, top=0.35)
correctoffset(aa1)

#profile2
tt2,aa2 = get_motion_pattern(A=1.35, T=0.6, N=20, top=0.35)
correctoffset(aa2)

#profile3
tt3,aa3 = get_motion_pattern(A=1.2, T=0.8, N=20, top=0.35)
correctoffset(aa3)



import visvis as vv

vv.figure(1); vv.clf();

a0 = vv.subplot(241); vv.title('profile0')
plot_pattern(*(tt0,aa0))
a0.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,1.3))
a0.axis.showGrid = True

a1 = vv.subplot(242); vv.title('profile1')
plot_pattern(*(tt1,aa1))
a1.axis.showGrid = True
a1.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,1.3))

a2 = vv.subplot(243); vv.title('profile2')
plot_pattern(*(tt2,aa2))
a2.axis.showGrid = True
a2.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,1.3))

a3 = vv.subplot(244); vv.title('profile3')
plot_pattern(*(tt3,aa3))
a3.axis.showGrid = True
a3.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,1.3))




## B series

#profile0
# tt0,aa0 = get_motion_pattern()
# correctoffset(aa0)

#profile1
tt1,aa1 = get_motion_pattern(A=0.6, T=0.8, N=20, top=0.35)
correctoffset(aa1)

#profile2
tt2,aa2 = get_motion_pattern(A=0.8, T=0.8, N=20, top=0.35)
correctoffset(aa2)

#profile3 - zit ook in A series
tt3,aa3 = get_motion_pattern(A=1.2, T=0.8, N=20, top=0.35)
correctoffset(aa3)

#profile4
tt4,aa4 = get_motion_pattern(A=2.0, T=0.8, N=20, top=0.35)
correctoffset(aa4)

#profile5
# extra = ampl, t peak top in perc T, sigma in perc T(a * G(t-b)_c)
tt5,aa5 = get_motion_pattern(A=2.0, T=0.8, N=20, top=0.35, extra=(0.7, 0.8, 0.05))
correctoffset(aa5)

#profile6
tt6,aa6 = get_motion_pattern(A=2.0, T=0.8, N=20, top=0.35, extra=(0.7, 0.8, 0.09)) 
correctoffset(aa6)

#profile7
# tt7,aa7 = get_motion_pattern() 
# correctoffset(aa7)


import visvis as vv

vv.figure(4); vv.clf();

# a0 = vv.subplot(241); vv.title('profile0')
# plot_pattern(*(tt0,aa0))
# a0.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))
# a0.axis.showGrid = True

a1 = vv.subplot(242); vv.title('profile1')
plot_pattern(*(tt1,aa1))
a1.axis.showGrid = True
a1.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

a2 = vv.subplot(243); vv.title('profile2')
plot_pattern(*(tt2,aa2))
a2.axis.showGrid = True
a2.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

a3 = vv.subplot(244); vv.title('profile3')
plot_pattern(*(tt3,aa3))
a3.axis.showGrid = True
a3.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

a4 = vv.subplot(245); vv.title('profile4')
plot_pattern(*(tt4,aa4))
a4.axis.showGrid = True
a4.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

a5 = vv.subplot(246); vv.title('profile5')
plot_pattern(*(tt5,aa5))
a5.axis.showGrid = True
a5.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

a6 = vv.subplot(247); vv.title('profile6')
plot_pattern(*(tt6,aa6))
a6.axis.showGrid = True
a6.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

# a7 = vv.subplot(248); vv.title('profile7')
# plot_pattern(*(tt7,aa7))
# a7.axis.showGrid = True
# a7.SetLimits(rangeX=(-1.2,1.2), rangeY=(-0.1,2))

## Store to excel

# patternToExcel(tt1, aa1, profile='profile1')



## Plot for paper, example
import os
from stentseg.utils.datahandling import select_dir
import matplotlib.pyplot as plt
import matplotlib as mpl
import prettyplotlib as ppl

f = vv.figure()
f.position = 0, 22, 1077, 273 
# B1  (A=0.6, T=0.8, N=20, top=0.35)
ax1 = vv.subplot(121)#; vv.title('profile1')
plot_pattern(*(tt1,aa1))
ax1.axis.showGrid = False
ax1.axis.showBox = False
ax1.SetLimits(rangeX=(-1,1), rangeY=(-0.02,2.5))
vv.xlabel('time (s)')
vv.ylabel('position (mm)')
vv.legend('B1 (A=0.6, T=0.8)')
# B5  (A=2.0, T=0.8, N=20, top=0.35, extra=(0.7, 0.8, 0.05))
ax5 = vv.subplot(122)#; vv.title('profile5')
plot_pattern(*(tt5,aa5))
ax5.axis.showGrid = False
ax5.SetLimits(rangeX=(-1,1), rangeY=(-0.02,2.5))
ax5.axis.showBox = False
vv.xlabel('time (s)')
vv.ylabel('position (mm)')
vv.legend('B5 (A=2.0, T=0.8, extra=0.7, 0.8, 0.05)')

f.relativeFontSize = 1.2

exceldir = select_dir(r'C:\Users\Maaike\Desktop',r'D:\Profiles\koenradesma\Desktop')
vv.screenshot(os.path.join(exceldir, 'screenshot.png'), f, sf=3, bg=(1,1,1))

## Plot for paper proceedings, example
import matplotlib.pyplot as plt
from lspeas.analysis.utils_analysis import _initaxis
from stentseg.utils.aortamotionpattern import plot_pattern_plt

dirsave = select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')

xlim = (0,2)
ylim = 2.5

f1 = plt.figure(figsize=(16,4.5), num=1); plt.clf()
ax0 = f1.add_subplot(121)
#todo: fix red line
plot_pattern_plt(*(tt1,aa1),label='B1 (A=0.6, T=0.8)')
_initaxis([ax0], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax0.set_ylim((-0.02, ylim))
ax0.set_xlim(xlim)

ax1 = f1.add_subplot(122)
plot_pattern_plt(*(tt5,aa5),label='B5 (A=2.0, T=0.8, extra=0.7, 0.8, 0.05)')
_initaxis([ax1], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax1.set_ylim((-0.02, ylim))
ax1.set_xlim(xlim)

# f1.savefig(os.path.join(dirsave, 'patternexampleproc.pdf'), papertype='a0', dpi=300)

