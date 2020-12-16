import args_kdiverse

args_kdiverse.dataset_ix = 5

import kfold_dataset_generator

fold_no = 5
kfold_dataset_generator.generate_ds_kfold_parts(5)
kfold_dataset_generator.generate_train_test_data(4, 5)
print("Fold No. : "+str(fold_no))
import kdiverse_generator

kdiverse_generator.generate_result(False, 3)
