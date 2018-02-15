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
        for phaserow in range(20,30): # excel rows 21-30
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
workbookF = 'Errors cam123ref_vs_alg Siemens.xlsx'
dirsave =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
savefig = True
# plot frequency profiles
# profiles, mean_abs_error, SD, MIN, Q1, Q3, MAX  = read_error_ouput(exceldir, workbookErrors)

profilesB = ['ZA6', 'ZB1', 'ZB2', 'ZB3', 'ZB6', 'ZB5', 'ZB4']
boxlabelsB = ['B0', 'B1', 'B2', 'B3', 'B6', 'B5', 'B4' ]
profilesA = [ 'ZA1', 'ZA2', 'ZA3']#, 'ZA3 STENT2']   # A1, 
boxlabelsA = ['A1', 'A2', 'A3']
profilesBxaxis = [0.02, 0.23, 0.37, 0.70, 1.22, 1.26, 1.36] # zb6=1.24
profilesAxaxis = [49.0, 96.3, 73.3]  

f1 = plt.figure(num=1, figsize=(14, 11)) # 9,5.5; 7.6,5
ax1 = f1.add_subplot(221)
ax2 = f1.add_subplot(222)

## amplitude patterns
def plot_ampl_errors(ax1,profilesBxaxis,abs_errors_profiles,meanabserrorsprofiles,maxabserrorprofiles):
    """
    """
    xlim = 1.45
    ylim = 0.67
    boxwidth = 0.06
    font = {'size': 14,
            }
    
    #plot line x=y
    ax1.plot([0,xlim],[0,xlim],'k--')
    ax1.plot([0,xlim],[0,xlim/2],'k-.')
    
    #plot text x=y
    ax1.text(0.78, 0.5, 'x=0.5y',fontdict=font)
    ax1.text(0.39, 0.5, 'x=y', fontdict=font)
    
    ax1.plot(profilesBxaxis, meanabserrorsprofiles, linestyle='', marker='.', color='b') 
    # ax1.plot(profilesBxaxis, maxabserrorprofiles, 'r--') # dotted line for min and max
    # meanvalues = np.asarray(meanabserrorsprofiles)
    # stdvalues = np.asarray(stdabserrorprofiles)
    # ax1.fill_between(profilesBxaxis, meanvalues-stdvalues,     
    #             meanvalues+stdvalues, color='r', alpha=0.2)
    
    # plot boxlabels
    boxypos = np.asarray(maxabserrorprofiles) + 0.015
    boxxpos = np.asarray(profilesBxaxis) - boxwidth/2.2
    for i, boxlabel in enumerate(boxlabelsB):
        ax1.text(boxxpos[i],boxypos[i], boxlabel, fontdict=font)
    
    bp = ax1.boxplot(abs_errors_profiles, 
                    positions=profilesBxaxis, 
                    widths=boxwidth,
                    patch_artist=True,  # fill with color) 
                    ) # dict
    
    # fill with colors
    gray = (211/255, 211/255, 211/255,  0.5) # with alpha
    colors = ['white', 'white', 'white', 'white', gray, gray, 'white']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    _initaxis([ax1], legend='upper right', xlabel='amplitude of reference pattern (mm)', 
            ylabel='absolute error (mm)')
    
    major_ticks = np.arange(0, xlim, 0.2)
    ax1.set_ylim((0, ylim))
    ax1.set_xlim(-0.04,xlim)
    ax1.set_xticks(major_ticks)
    # plt.setp(bp['boxes'], color='k')
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], color='k', marker='+')
    # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    
    return bp

# get errors
abs_errors_profiles = read_error_cam123(exceldir, workbook, profilesB)

meanabserrorsprofiles = [np.mean(x) for x in abs_errors_profiles]
stdabserrorprofiles = [np.std(x) for x in abs_errors_profiles]
maxabserrorprofiles = [np.max(x) for x in abs_errors_profiles]
q75abserrorprofiles = [np.percentile(x,75) for x in abs_errors_profiles]

abs_errors_profilesF = read_error_cam123(exceldir, workbookF, profilesB)

meanabserrorsprofilesF = [np.mean(x) for x in abs_errors_profilesF]
stdabserrorprofilesF = [np.std(x) for x in abs_errors_profilesF]
maxabserrorprofilesF = [np.max(x) for x in abs_errors_profilesF]
q75abserrorprofilesF = [np.percentile(x,75) for x in abs_errors_profilesF]


# Toshiba
bpTa = plot_ampl_errors(ax1,profilesBxaxis,abs_errors_profiles,meanabserrorsprofiles,maxabserrorprofiles)
# Flash
bpFa = plot_ampl_errors(ax2,profilesBxaxis,abs_errors_profilesF,meanabserrorsprofilesF,maxabserrorprofilesF)


# # save
# if savefig:
#     f1.savefig(os.path.join(dirsave, 'abserrorgraphampl.pdf'), papertype='a0', dpi=600)


## frequency patterns
abs_errors_profiles = read_error_cam123(exceldir, workbook, profilesA)

#Tosh
meanabserrorsprofiles = [np.mean(x) for x in abs_errors_profiles]
stdabserrorprofiles = [np.std(x) for x in abs_errors_profiles]
maxabserrorprofiles = [np.max(x) for x in abs_errors_profiles]
q75abserrorprofiles = [np.percentile(x,75) for x in abs_errors_profiles]

#Flash
abs_errors_profilesF = read_error_cam123(exceldir, workbookF, profilesA)

meanabserrorsprofilesF = [np.mean(x) for x in abs_errors_profilesF]
stdabserrorprofilesF = [np.std(x) for x in abs_errors_profilesF]
maxabserrorprofilesF = [np.max(x) for x in abs_errors_profilesF]
q75abserrorprofilesF = [np.percentile(x,75) for x in abs_errors_profilesF]


# f1 = plt.figure(num=2, figsize=(7.6, 5))
# ax3 = f1.add_subplot(111)
ax3 = f1.add_subplot(223)
ax4 = f1.add_subplot(224)


def plot_freq_errors(ax3,profilesAxaxis,abs_errors_profiles,meanabserrorsprofiles,maxabserrorprofiles):
    """
    """
    ylim = 0.67
    boxwidth = 2.2
    font = {'size': 14,
            }
    
    ax3.plot(profilesAxaxis, meanabserrorsprofiles, linestyle='', marker='.', color='b') 
    # ax3.plot(profilesAxaxis, maxabserrorprofiles, 'r--') # dotted line for min and max
    
    # plot boxlabels
    boxypos = np.asarray(maxabserrorprofiles) + 0.015
    boxxpos = np.asarray(profilesAxaxis) - boxwidth/2.2
    for i, boxlabel in enumerate(boxlabelsA):
        ax3.text(boxxpos[i],boxypos[i], boxlabel, fontdict=font)
    
    bp = ax3.boxplot(abs_errors_profiles, 
                    positions=profilesAxaxis, 
                    widths=boxwidth,
                    )
    
    _initaxis([ax3], legend='upper right', xlabel='frequency of reference pattern (bpm)', 
            ylabel='absolute error (mm)')
    major_ticks = np.arange(40, 100, 10)
    ax3.set_ylim((0, ylim))
    ax3.set_xlim(40,100)
    ax3.set_xticks(major_ticks)
    plt.setp(bp['boxes'], color='k')
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], color='k', marker='+')
    plt.setp(bp['medians'], color='k' )
    
    return bp

bpTf = plot_freq_errors(ax3,profilesAxaxis,abs_errors_profiles,meanabserrorsprofiles,maxabserrorprofiles)
bpFf = plot_freq_errors(ax4,profilesAxaxis,abs_errors_profilesF,meanabserrorsprofilesF,maxabserrorprofilesF)


# save
if savefig:
    f1.savefig(os.path.join(dirsave, 'abserrorgraphamplfreq.pdf'), papertype='a0', dpi=600)





