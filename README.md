# MolDVC-DTA
---
A repo for "MolDVC-DTA: Molecular Structural Dual-View Learning with Contrastive Enhancement for Drug-Target Affinity Prediction".



## Abstracts

Drug-target affinity (DTA) prediction is crucial in drug discovery as it effectively shortens the time and reduces the cost of drug development. In recent years, deep learning models based on drug topological graphs have been widely applied to DTA prediction tasks. However, these methods often rely on a singular atomic perspective and overlook the critical role of chemical bond information in drug molecules. Exploring rich and complete structural information of drugs and deeply mining molecular representations are key to improving DTA prediction accuracy. Therefore, we propose MolDVC-DTA, a dual-view contrastive-enhanced deep learning model based on molecular structure. This model utilizes a dual-view architecture to deeply integrate atomic and chemical bond topological information, providing a comprehensive and accurate description of drug molecules. In addition, we have developed a Multi-scale Representation Extraction and Contrastive Learning Enhancer to precisely identify key substructures of drug molecules and achieve feature reinforcement. Comprehensive experimental results demonstrate that MolDVC-DTA outperforms current state-of-the-art (SOTA) methods on benchmark datasets for DTA tasks. We further validated the generalization ability of MolDVC-DTA in classification tasks and demonstrated the interpretability of the method through visual analysis.



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
  

## Results

* ### Experimental results

  We have designed a protein semantic information fusion framework based on the concept of hierarchical graph to enhance the richness of protein representation. Meanwhile, we propose two different strategies for semantic information fusion (_Top-Down_ and _Bottom-Up_) and evaluate their performance on different datasets. The performance of two different strategies on different datasets is as follows:

  1. __Performance on the Davis dataset__
     <a name="Experimental results on davis dataset"></a>
  
      | Backbone | MSE          | CI |
      |:--------:|:---------:|:--------------:|
      | __TDNet__ (Top-Down)    |  0.193 | 0.907  |
      | __BUNet__ (Bottom-Up)   |  0.191 | 0.906 |

  2. __Performance on the KIBA dataset__
      <a name="Experimental results on kiba dataset"></a>
        
      | Backbone | MSE          | CI |
      |:--------:|:---------:|:--------------:|
      | __TDNet__ (Top-Down) |  0.120 | 0.904 |
      | __BUNet__ (Bottom-Up)|  0.121 | 0.904 |

  3. __Performance on the Human dataset__
      <a name="Experimental results on kiba dataset"></a>
     
      | Backbone | AUROC     | Precision |  Recall |
      |:--------:|:---------:|:--------------:|:-------:|
      | __TDNet__ (Top-Down) |  0.988 | 0.945 | 0.952 |
      | __BUNet__ (Bottom-Up)|  0.986 | 0.947 | 0.947|

   🌳 The performance of baseline models can be found in `experimental_results.ipynb` or `baselines` directory.
   
* ### Reproduce the results with single command
   To facilitate the reproducibility of our experimental results, we have provided a Docker Image-based solution that allows for reproducing our experimental results on multiple datasets with just a single command. You can easily experience this function with the following simple command：
  ```text
  sudo docker run --name hisif-con --gpus all --shm-size=2g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1

  # docker run ：Create and start a new container based on the specified image.
  # --name : It specifies the name ("hisif-con") for the container being created. You can use this name to reference and manage the container later.
  # --gpus : It enables GPU support within the container and assigns all available GPUs to it. This allows the container to utilize the GPU resources for computation.
  # -v : This is a parameter used to map local files to the container,and it is used in the following format: `-v /your/local/path/HiSIF-DTA:/mapped/container/path/HiSIF-DTA`
  # -it : These options are combined and used to allocate a pseudo-TTY and enable interactive mode, allowing the user to interact with the container's command-line interface.
  # hisif-image:v1 : It is a doker image, builded from Dockerfile. For detailed build instructions, please refer to the `Requirements` section.
  ```
  :bulb: Please note that the above one-click run is only applicable for the inference process and requires you to pre-place all the necessary processed data and pretrained models in the correct locations on your local machine. If you want to train the model in the created Docker container, please follow the instructions below:
   ```text
   1. sudo docker run --name hisif-con --gpus all --shm-size=16g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1 /bin/bash
   2. cd /media/HiSIF-DTA
   3. python training_for_DTA.py --dataset davis --model TDNet
   ```
## Baseline models

To demonstrate the superiority of the proposed model, we conduct experiments to compare our approach with the following state-of-the-art (SOTA) models:

**DTA:**
- **DeepDTA** : [Repo Link ](https://github.com/hkmztrk/DeepDTA)
- **AttentionDTA** : [Repo Link ](https://github.com/zhaoqichang/AttentionDTA_BIBM)
- **GraphDTA** : [Repo Link ](https://github.com/thinng/GraphDTA)
- **MGraphDTA** : [Repo Link ](https://github.com/guaguabujianle/MGraphDTA)
- **DGraphDTA** : [Repo Link ](https://github.com/595693085/DGraphDTA)

**CPI:**
- **DrugVQA (seq)** : [Repo Link ](https://github.com/prokia/drugVQA)
- **GraphDTA** : [Repo Link ](https://github.com/thinng/GraphDTA)
- **CPI-GNN** : [Repo Link ](https://github.com/masashitsubaki/CPI_prediction)
- **TransformerCPI** : [Repo Link ](https://github.com/lifanchen-simm/transformerCPI)

🌳 The above link is the GitHub link to the baseline models. To ensure a fair comparison, we re-trained these baseline models with the same experimental setup as our proposed model. The detailed re-training codes and results can be found in the `baselines` directory.

## NoteBooks

To ensure the transparency of experimental results, the prediction results of all models (including our proposed model and baseline models) have been uploaded to Zenodo ([Link](https://zenodo.org/record/8385073)). Additionally, in order to present the experimental results in a more intuitive way, we provide a comprehensive Jupyter notebook in our repo ([`experimental_results.ipynb`](https://github.com/bixiangpeng/HiSIF-DTA/blob/main/experimental_results.ipynb)), where we load all prediction result files and recalculate the experimental metrics based on these results, presenting them in the form of statistical charts or tables.

## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.
