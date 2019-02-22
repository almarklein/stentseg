""" Read position errors from excel for statistical analysis

"""
import os
from stentseg.utils.datahandling import select_dir
import openpyxl # http://openpyxl.readthedocs.org/
import numpy as np
from lspeas.utils.normality_statistics import paired_samples_ttest


def read_error_cam123(exceldir, workbook, profiles):
    """ read the absolute errors for 10 timepositions for all stent points
    """
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbook), data_only=True)
    abs_errors_profiles = []
    for profile in profiles:
        sheet = wb.get_sheet_by_name(profile)
        abs_errors_profile = []
        for phaserow in range(8,18): # excel rows 21-30 when 20,30; rows 9-18 when 8,18
            abs_errors = sheet.rows[phaserow][1:] # skip first col with notes
            abs_errors = [obj.value for obj in abs_errors if obj.value is not None]
            abs_errors_profile.append(abs_errors)
        spread = np.concatenate([a for a in abs_errors_profile], axis=0)
        abs_errors_profiles.append(spread)
    
    return abs_errors_profiles

def read_ampl_errorcam123(exceldir, workbook, profile):
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbook), data_only=True)
    sheet = wb.get_sheet_by_name(profile)
    phaserow = 58 - 1 # 58 for 58; 60 for 60
    errors = sheet.rows[phaserow][1:] # skip first col with notes
    errors = [obj.value for obj in errors if obj.value is not None]
    return errors


exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot',
                      r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot')
workbook = 'Errors cam123ref_vs_alg Toshiba.xlsx'
workbookF = 'Errors cam123ref_vs_alg Siemens.xlsx'

## test over all 10 positions
prof = 'ZA1'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZA2'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZA3'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZA6'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB1'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB2'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB3'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB4'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB5'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


prof = 'ZB6'
abs_errors_T_x = read_error_cam123(exceldir, workbook, [prof])
abs_errors_F_x = read_error_cam123(exceldir, workbookF, [prof])

t2, p2 = paired_samples_ttest(abs_errors_T_x, abs_errors_F_x, prof)


## for the amplitudes
print("******* Amplitude errors *********")

prof = 'ZA1'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZA2'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZA3'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZA6'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB1'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB2'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB3'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB4'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB5'
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)


prof = 'ZB6' 
errors_T_x = read_ampl_errorcam123(exceldir, workbook, prof)
errors_F_x = read_ampl_errorcam123(exceldir, workbookF, prof)

t2, p2 = paired_samples_ttest(errors_T_x, errors_F_x, prof, amplitude=True)