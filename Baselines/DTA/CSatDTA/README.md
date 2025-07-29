# CSat-DTA
This is python based project. the title stands for Convolution with Self Attention based Drug Target Affinity  To run the project, please run main.py

## RUN

#### KiBA
```
python main.py --num_windows 32 --seq_window_lengths 8 12 --smi_window_lengths 4 8 --batch_size 64 --num_epoch 200 --max_seq_len 1000 --max_smi_len 100 --dataset_path data/kiba/ --problem_type 1 --log_dir 'logs/'
```
#### Davis
```
python main.py --num_windows 32 --seq_window_lengths 8 12 --smi_window_lengths 4 8 --batch_size 64 --num_epoch 200 --max_seq_len 1000 --max_smi_len 100 --dataset_path data/davis/ --problem_type 1 --is_log 1 --log_dir 'logs/'
```