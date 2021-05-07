import args_kdiverse
import os
import glob
import zipfile

if os.path.exists('model_files'):
    files = glob.glob('model_files/*')
    for f in files:
        os.remove(f)

if not os.path.exists('recset_nasr'):
    os.mkdir('recset_nasr')
else:
    files = glob.glob('recset_nasr/*')
    for f in files:
        os.remove(f)

args_kdiverse.dat_ix = 2
args_kdiverse.FOLD = 5
args_kdiverse.test_index = 1
args_kdiverse.copy_no = 0

K = 5

model_zip_name = 'model_files_ds_' + str(args_kdiverse.dat_ix) + '_index_' \
                 + str(args_kdiverse.test_index) + '.zip'

with zipfile.ZipFile(os.path.join('model_repository', model_zip_name), 'r') as zip_ref:
    zip_ref.extractall('.')

from kfold_dataset_generator import generate_ds

generate_ds(args_kdiverse.dat_ix, args_kdiverse.FOLD, args_kdiverse.test_index, args_kdiverse.copy_no)

from kdiverse_generator import generate_result

Ns = [(5, 5)]

for Nmn, Nmx in Ns:
    generate_result(True, K=K, N_min=Nmn, N_max=Nmx)
