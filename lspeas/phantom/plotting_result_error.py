""" Plotting result of motion_pattern_error

"""


def read_error_ouput(exceldir, workbook, rowS=18, colS=1, colE=5):
    """
    """
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbook), data_only=True)
    sheet = wb.get_sheet_by_name('summery')
    rowS = rowS
    colS = colS
    colE = colE
    mean_abs_error = sheet.rows[rowS][colS:colE]
    mean_abs_error = [obj.value for obj in mean_abs_error] 
    SD = sheet.rows[rowS+1][colS:colE]
    SD = [obj.value for obj in SD] 
    MIN = sheet.rows[rowS+2][colS:colE]
    MIN = [obj.value for obj in MIN] 
    Q1 = sheet.rows[rowS+3][colS:colE]
    Q1 = [obj.value for obj in Q1] 
    Q3 = sheet.rows[rowS+4][colS:colE]
    Q3 = [obj.value for obj in Q3] 
    MAX = sheet.rows[rowS+5][colS:colE]
    MAX = [obj.value for obj in MAX]
    profiles = sheet.rows[7][colS:colE]
    profiles = [obj.value for obj in profiles]
    
    return profiles, mean_abs_error, SD, MIN, Q1, Q3, MAX 
    


import os
import openpyxl
import matplotlib.pyplot as plt
from stentseg.utils.datahandling import select_dir
# import seaborn as sns  #sns.tsplot
# https://www.wakari.io/sharing/bundle/ijstokes/pyvis-1h?has_login=False
# http://spartanideas.msu.edu/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/

exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot',
                      r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot')
workbookErrors = 'Errors camera_algorithm Toshiba.xlsx'
dirsave =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')

# plot frequency profiles
profiles, mean_abs_error, SD, MIN, Q1, Q3, MAX  = read_error_ouput(exceldir, workbookErrors)

f1 = plt.figure(num=1, figsize=(7.6, 5))
ax1 = f1.add_subplot(111)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left()
ax1.plot(profiles, mean_abs_error, linestyle='', marker='o', color='b') 
ax1.errorbar(profiles, mean_abs_error, yerr = SD, fmt=None, color='b', capsize=8)
# plt.xticks(range(len(mean_abs_error)), profiles, size = 'medium')
ax1.set_xlabel('heart rate (bpm)', fontsize=14)
ax1.set_ylabel('absolute error (mm)', fontsize=14)
plt.xlim(45,105)
plt.ylim(0,0.3)
# save
plt.savefig(os.path.join(dirsave, 'errorgraphfreq.pdf'), papertype='a0', dpi=300)


# plot amplitude profiles
profiles, mean_abs_error, SD, MIN, Q1, Q3, MAX  = read_error_ouput(exceldir, workbookErrors, colS=5, colE=12)

f2 = plt.figure(num=3, figsize=(7.6, 5))
ax2 = f2.add_subplot(111)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)
ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left()
ax2.plot(profiles[0], mean_abs_error[0], linestyle='', marker='o', color='k') 
ax2.errorbar(profiles[0], mean_abs_error[0], yerr = SD[0], fmt=None, ecolor='k', capsize=8)
ax2.plot(profiles[1:-2], mean_abs_error[1:-2], linestyle='', marker='o', color='b') 
ax2.errorbar(profiles[1:-2], mean_abs_error[1:-2], yerr = SD[1:-2], fmt=None, ecolor='b', capsize=8)
ax2.plot(profiles[-2:], mean_abs_error[-2:], linestyle='', marker='o', color='r')
ax2.errorbar(profiles[-2:], mean_abs_error[-2:], yerr = SD[-2:], fmt=None, ecolor='r', capsize=8) 
# ax2.plot(profiles, Q1, 'b.--')
# ax2.plot(profiles, Q3, 'b.--')
# plt.xticks(range(len(mean_abs_error)), profiles, size = 'medium')
ax2.set_xlabel('amplitude (mm)', fontsize=14)
ax2.set_ylabel('absolute error (mm)', fontsize=14)
plt.xlim(0,1.45)
plt.ylim(0,0.3)
# save
plt.savefig(os.path.join(dirsave, 'errorgraphampl.pdf'), papertype='a0', dpi=300)