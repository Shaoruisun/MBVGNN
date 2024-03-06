# Quick Start

## 1. download the cif file to the folder "./database/cif"

## 2. run Graph.py to creat the graph input dataset of the model

e.g. run:

python Graph.py --data_dir ./database/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1

## 3. run train_MBVGNN.py to train the MBVGNN model

  e.g. run:  
  
python train_MBVGNN.py --n_hidden_feat 216 --n_GCN_feat 216 --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs 1000 --batch_size 30 --target_name average_voltage --milestones 900 --gamma 0.1 --test_ratio 0.1 --datafile_name my_graph_data_MP_test1_8_12_100_000 --database Eg --n_Gaussian 128 --N_block 3 --lr 8e-4 &
