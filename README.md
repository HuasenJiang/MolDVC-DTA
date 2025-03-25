# MolDVC-DTA
---



![Figure_1](https://github.com/user-attachments/assets/7253788d-fa2a-480e-8571-aa298d633433)

## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `https://github.com/HuasenJiang/MolDVC-DTA`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n MolDVC python=3.8 ` and `conda activate MolDVC`
   Install specified version of pytorch: `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
  
   torch-geometric == 2.4.0
  
   torch-cluster == 1.6.1
  
   torch-scatter == 2.1.1
  
   torch-sparse == 0.6.17
  
   torch-spline-conv == 1.2.2
  
   rdkit == 2023.9.5
  
   pandas == 2.0.3
  
   scikit-learn == 1.3.2
  
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `23.3.1`.

  
##  Usages

* ### Data preparation
  There are three benchmark datasets were adopted in this project, including two DTA datasets (`Davis and KIBA`) and a CPI dataset (`Human`).

   1. __Download processed data__
   
      The data file (`data.zip`) of these three datasets can be downloaded from this [link](https://pan.baidu.com/s/1VvKdQQzl1vbHcVw9URvxLA?pwd=1234 ). Uncompress this file to get a 'data' folder containing all the original data and processed data.
      
      🌳 Replacing the original 'data' folder by this new folder and then you can re-train or test our proposed model on Davis, KIBA or Human.  
      
      🌳 For clarity, the file architecture of `data` directory is described as follows:
      
      ```text
       >  data
          ├── davis / kiba                          - DTA dataset directory.
          │   ├── ligands_can.txt                   - A txt file recording ligands information (Original)
          │   ├── proteins.txt                      - A txt file recording proteins information (Original)
          │   ├── Y                                 - A file recording binding affinity score (Original)
          │   ├── folds                         
          │   │   ├── test_fold_setting1.txt        - A txt file recording test set entry (Original)
          │   │   └── train_fold_setting1.txt       - A txt file recording training set entry (Original)
          │   ├── (davis/kiba)_dict.txt             - A txt file recording the corresponding Uniprot ID for every protein in datasets (processed)
          │   ├── train.csv                         - Training set data in CSV format (processed)
          │   ├── test.csv                          - Test set data in CSV format (processed)
          │   ├── mol_data_M.pkl                      - A pkl file recording drug graph data for all drugs in dataset (processed)
          │   └── pro_data_M.pkl                      - A pkl file recording protein graph data for all proteins in dataset (processed)
          └── Human                                 - CPI dataset directory.
              ├── Human.txt                         - A txt file recording the information of drugs and proteins that interact (Original)              
              ├── Human_dict.txt
              ├── train(fold).csv                   - 5-fold training set data in CSV format (processed)
              ├── test(fold).csv                    - 5-fold test set data in CSV format (processed)
              ├── mol_data_M.pkl
              └── pro_data_M.pkl
      ```
   3. __Customize your data__

      You might like to test the model on more DTA or CPI datasets. If this is the case, please add your data in the folder 'data' and process them to be suitable for our model. We provide a detailed processing script for converting original data to the input data that our model needed, i.e., `create_data.py`. The processing steps are as follows:
     
      1. Split the raw dataset into training and test sets, and convert them into CSV format respectively（i.e., `train.csv` and `test.csv`）.
         The content of the csv file can be organized as follows:
         ```text
                   compound_iso_smiles                                 target_sequence                                       affinity
         C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1          MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSY                        5.568636236
                                                             YEYDFERGRRGSKKGSIDVEKITCVETVVPEKNPPPERQ
                                                             IPRRGEESSEMEQISIIERFPYPFQVVYDEGP
         ```
      2. Collect the Uniprot ID of all proteins in dataset from Uniprot DB(https://www.uniprot.org/) and record it as a txt file, such as `davis_dict.txt`:
         ```text
         >MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILM...	Q2M2I8
         >PFWKILNPLLERGTYYYFMGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVEHEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLR...	P00519
  
      3. Construct the graph data for drugs and seq data proteins. Assume that you already have aboving files (1.2.3) in your `data/your_dataset_name/` folder, you can simply run following scripts:
         ```python
         python created_data.py --path '..data/'  --dataset 'your_dataset_name'  --output_path '..data/'
         ```
      
   :bulb: Note that the above is just a description of the general steps, and you may need to make some modification to the original script for different datasets.
     
   :blush: Therefore，We have provided detailed comments on the functionality of each function in the script, hoping that it could be helpful for you.

* ### Training
  After processing the data, you can retrain the model from scratch with the following command:
  ```text
  
  python training_for_DTA.py  --epochs 2000 --batch 512 --LR 0.0005 --log_interval 20 --device 0 --dataset davis/kiba --num_workers 8
  or
  python training_for_CPI.py  --epochs 2000 --batch 512 --LR 0.0005 --log_interval 20 --device 0 --dataset Human --num_workers 8
  ```
   Here is the detailed introduction of the optional parameters when running `training_for_DTA/CPI.py`:
     ```text
      --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
      --batch: The batch size, specifying the number of samples in each training batch.
      --LR: The learning rate, controlling the rate at which model parameters are updated.
      --log_interval: The log interval, specifying the time interval for printing logs during training.
      --device: The device, specifying the GPU device number used for training.
      --dataset: The dataset name, specifying the dataset used for model training.
      --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables 
       multiprocessing for data processing.
   ```
   🌳 We provided an additional training file (`training_for_CPI.py`) specifically for conducting five-fold cross-training on the Human dataset.
  
## Contact

We welcome you to contact us (email: Jianghuasen@stu.ouc.edu.cn) for any questions and cooperations.
