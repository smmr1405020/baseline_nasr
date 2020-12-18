import os
if not os.path.exists('gtset_nasr'):
    os.mkdir('gtset_nasr')
if not os.path.exists('recset_nasr'):
    os.mkdir('recset_nasr')

import args_kdiverse

args_kdiverse.dataset_ix = 7

import kfold_dataset_generator

fold_no = 1
kfold_dataset_generator.generate_ds_kfold_parts(5)
kfold_dataset_generator.generate_train_test_data(fold_no, 5)
print("Fold No. : "+str(fold_no))
import kdiverse_generator

kdiverse_generator.generate_result(False, 3)
