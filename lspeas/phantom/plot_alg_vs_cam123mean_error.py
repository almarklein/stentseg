""" Plotting result of alg_vs_cam123mean

"""

def read_error_cam123(exceldir, workbook, profiles):
    """ read the absolute errors for 10 timepositions for all stent points
    """
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbook), data_only=True)
    abs_errors_profiles = []
    for profile in profiles:
        sheet = wb.get_sheet_by_name(profile)
        abs_errors_profile = []
        for phaserow in range(20,30): # 21-30
            abs_errors = sheet.rows[phaserow]
            abs_errors = [obj.value for obj in abs_errors if obj.value is not None]
            abs_errors_profile.append(abs_errors)
        spread = np.concatenate([a for a in abs_errors_profile], axis=0)
        abs_errors_profiles.append(spread)
    
    return abs_errors_profiles
    


import os
import openpyxl
import matplotlib.pyplot as plt
from stentseg.utils.datahandling import select_dir
import numpy as np
from lspeas.analysis.utils_analysis import _initaxis
# import seaborn as sns  #sns.tsplot
# https://www.wakari.io/sharing/bundle/ijstokes/pyvis-1h?has_login=False
# http://spartanideas.msu.edu/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/

exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot',
                      r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot')
workbook = 'Errors cam123ref_vs_alg Toshiba.xlsx'
dirsave =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
savefig = False
# plot frequency profiles
# profiles, mean_abs_error, SD, MIN, Q1, Q3, MAX  = read_error_ouput(exceldir, workbookErrors)

profilesB = ['ZA6', 'ZB1', 'ZB2', 'ZB3', 'ZB6', 'ZB5', 'ZB4']
profilesA = [ 'ZA1', 'ZA2', 'ZA3']#, 'ZA3 STENT2']   # A1, 
profilesBxaxis = [0.02, 0.23, 0.37,0.71, 1.19, 1.26, 1.36]
profilesAxaxis = [49.0, 96.3, 73.3]  

## amplitude patterns
abs_errors_profiles = read_error_cam123(exceldir, workbook, profilesB)

meanabserrorsprofiles = [np.mean(x) for x in abs_errors_profiles]
stdabserrorprofiles = [np.std(x) for x in abs_errors_profiles]
maxabserrorprofiles = [np.max(x) for x in abs_errors_profiles]
q75abserrorprofiles = [np.percentile(x,75) for x in abs_errors_profiles]


f1 = plt.figure(num=1, figsize=(7.6, 5))
ax1 = f1.add_subplot(111)
xlim = 1.45
ylim = 0.65

ax1.plot(profilesBxaxis, meanabserrorsprofiles, linestyle='r-', marker='.', color='b') 
# ax1.plot(profilesBxaxis, maxabserrorprofiles, 'r--') # dotted line for min and max
# meanvalues = np.asarray(meanabserrorsprofiles)
# stdvalues = np.asarray(stdabserrorprofiles)
# ax1.fill_between(profilesBxaxis, meanvalues-stdvalues,     
#             meanvalues+stdvalues, color='r', alpha=0.2)
bp = plt.boxplot(abs_errors_profiles, positions=profilesBxaxis, widths=0.06)

_initaxis([ax1], legend='upper right', xlabel='amplitude of pattern (mm)', 
          ylabel='absolute error (mm)')
major_ticks = np.arange(0, xlim, 0.2)
ax1.set_ylim((0, ylim))
ax1.set_xlim(-0.04,xlim)
ax1.set_xticks(major_ticks)
plt.setp(bp['boxes'], color='k')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], color='k', marker='+')

# save
if savefig:
    f1.savefig(os.path.join(dirsave, 'abserrorgraphampl.pdf'), papertype='a0', dpi=300)


## frequency patterns
abs_errors_profiles = read_error_cam123(exceldir, workbook, profilesA)

meanabserrorsprofiles = [np.mean(x) for x in abs_errors_profiles]
stdabserrorprofiles = [np.std(x) for x in abs_errors_profiles]
maxabserrorprofiles = [np.max(x) for x in abs_errors_profiles]
q75abserrorprofiles = [np.percentile(x,75) for x in abs_errors_profiles]


f1 = plt.figure(num=2, figsize=(7.6, 5))
ax1 = f1.add_subplot(111)
ylim = 0.65

ax1.plot(profilesAxaxis, meanabserrorsprofiles, linestyle='r-', marker='.', color='b') 
# ax1.plot(profilesAxaxis, maxabserrorprofiles, 'r--') # dotted line for min and max

bp = plt.boxplot(abs_errors_profiles, positions=profilesAxaxis, widths=2.5)

_initaxis([ax1], legend='upper right', xlabel='frequency of pattern (bpm)', 
          ylabel='absolute error (mm)')
major_ticks = np.arange(40, 100, 10)
ax1.set_ylim((0, ylim))
ax1.set_xlim(40,100)
ax1.set_xticks(major_ticks)
plt.setp(bp['boxes'], color='k')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], color='k', marker='+')

# save
if savefig:
    f1.savefig(os.path.join(dirsave, 'abserrorgraphfreq.pdf'), papertype='a0', dpi=300)





