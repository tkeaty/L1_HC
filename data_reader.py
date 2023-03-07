import pandas as pd
import os
import numpy as np


def check_case(case, type):
    if case:
        if 'cancer' in type.lower():
            return True
        return False

    else:
        if 'normal' in type.lower():
            return True
        return False


def create_dataframe(d, s_file, out_file, case=False):
    """
    Function to create dataframe from TCGA files
    :param d: The directory storing the folders
    :param s_file: The filename for the Sample Sheet
    :param out_file: Output filename
    :param case: Set to true if creating case data
    """
    c = True
    # Check
    luad_samples = pd.read_csv(d + s_file, delimiter='\t')
    type_dict = {}
    id_dict = {}
    for id, type, sid in zip(luad_samples['File ID'].values, luad_samples['Sample Type'].values, luad_samples['Case ID'].values):
        type_dict[id] = type
        id_dict[id] = sid

    cols = []
    expr_data = []
    sample_ids = []
    for subdir, dirs, files in os.walk(d):
        if subdir == d:
            continue
        else:
            id = subdir[10:]
            # Check
            if check_case(case, type_dict[id]):
                sample_ids.append(id_dict[id].split('-')[1])
                for file in files:
                    if file[-4:] != '.tsv':
                        continue

                    with open(os.path.join(subdir, file), 'rt') as f:
                        data = []
                        i = 0
                        for line in f:
                            if i == 0:
                                i += 1
                            elif i < 6:
                                i += 1
                            else:
                                temp = line.strip().split('\t')
                                data.append([temp[1], temp[3]])

                        data_arr = np.asarray(data).T
                        expr_data.append(list(data_arr[1,:]))
                        if c:
                            cols = list(data_arr[0,:])
                            c = False

    df = pd.DataFrame(expr_data, columns=cols)
    print(df.shape[0])
    df.apply(pd.to_numeric)
    # Check
    df['Batch'] = sample_ids
    df.to_csv(d + out_file)


def re_batch(case, ctrl=None):
    if ctrl is None:
        batches = np.unique(case['Batch'].values)
        batch_cts = {b: case[case['Batch']==b].shape[0] for b in batches}
        print(batch_cts)



# all_case = pd.read_csv('ALL_files/ALL_case_counts_batch.csv')
# print(all_case['Batch'].head())
# re_batch(all_case)

from scipy.stats import chi2

print(chi2.pdf(85.97, df=1))

# create_dataframe(d, s_file, out_file, case=False)
# create_dataframe('ALL_files/', 'gdc_sample_sheet.2022-04-17.tsv', 'ALL_case_counts_batch.csv', case=True)
# create_dataframe('AML_files/', 'gdc_sample_sheet.2022-04-17.tsv', 'AML_ctrl_counts_batch.csv', case=False)
# create_dataframe('AML_files/', 'gdc_sample_sheet.2022-04-17.tsv', 'AML_case_counts_batch.csv', case=True)
