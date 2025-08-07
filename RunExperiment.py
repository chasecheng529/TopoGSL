import os
import torch
import random
import math
from Pretrain import Pretrain
from Arguments import argparser
from FineTuning import FineTune
from rdkit import RDLogger
import torch.nn.functional as F


# 禁用 RDKit 的日志
RDLogger.DisableLog('rdApp.*')

def FineTuneMultiDownstreamTasks(ARGS, torchSeed):
    downstreamTaskPerform = {}
    for eachTask in ARGS.DownstreamTasks:
        torch.manual_seed(torchSeed)
        if eachTask == "BBBP":
            taskType = "classification"
            taskNum = 1
            tasklist=["p_np"]

        elif eachTask == "BACE":
            taskType = "classification"
            taskNum = 1
            tasklist=["Class"]

        elif eachTask == "HIV":
            taskType = "classification"
            taskNum = 1
            tasklist=["HIV_active"]   

        elif eachTask == "ClinTox":
            taskType = "classification"
            taskNum = 2
            tasklist=['CT_TOX', 'FDA_APPROVED']

        elif eachTask == "SIDER":
            taskType = "classification"
            taskNum = 27
            tasklist=[
                "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
                "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
                "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
                "Reproductive system and breast disorders", 
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
                "General disorders and administration site conditions", "Endocrine disorders", 
                "Surgical and medical procedures", "Vascular disorders", 
                "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
                "Congenital, familial and genetic disorders", "Infections and infestations", 
                "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
                "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
                "Ear and labyrinth disorders", "Cardiac disorders", 
                "Nervous system disorders", "Injury, poisoning and procedural complications"
            ]

        elif eachTask == "TOX21":
            taskType = "classification"
            taskNum = 12
            tasklist=[
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
            ]

        elif eachTask == "ESOL":
            taskType = "RMSE"
            taskNum = 1
            tasklist=["measured log solubility in mols per litre"]

        elif eachTask == "FreeSolv":
            taskType = "RMSE"
            taskNum = 1
            tasklist=["expt"]

        elif eachTask == "Lipo":
            taskType = "RMSE"
            taskNum = 1
            tasklist=["exp"]

        elif eachTask == "QM7":
            taskType = "MAE"
            taskNum = 1
            tasklist=["u0_atom"]

        elif eachTask == "QM8":
            taskType = "MAE"
            taskNum = 12
            tasklist=[
                "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
                "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"]
        else:
            print("Error! Unexcpted Task.")
            exit()
        downstreamTaskPerform[eachTask] = FineTune(ARGS, eachTask, taskType, tasklist, taskNum)
    return downstreamTaskPerform

def PrintARGSInfo(ARGS):
    print("========================ARGS======================")
    print("git: {}, task pid:{}".format(ARGS.gitNode, ARGS.PID))
    print("num features: {}, run type: {}".format(ARGS.FeatureNums, ARGS.runType))
    print("============Pretrain ARGS============")
    print("pretrain epoch: {}, pretrain warmup: {}".format(ARGS.pretrainEpoch, ARGS.pretrainWarmUp))
    print("pretrain batch size: {}, pretrain config: {}".format(ARGS.pretrainBatchSize, ARGS.pretrainConfig))
    print("============Finetuning ARGS==========")
    print("chechpoint name: {}, tuning epoch: {}".format(ARGS.checkPointName, ARGS.TuningEpoch))
    print("tuning batch size: {}, tuning config: {}".format(ARGS.TuneBatchSize, ARGS.tuningConfig))
    print("========================ARGS======================")

def GetMultiTuningResult(dict_list):
    if not dict_list:
        return {}

    # 初始化累加器
    sum_dict = {}
    sumsq_dict = {}
    count = len(dict_list)

    for d in dict_list:
        for k, v in d.items():
            sum_dict[k] = sum_dict.get(k, 0.0) + v
            sumsq_dict[k] = sumsq_dict.get(k, 0.0) + v * v

    # 计算平均值和标准差
    result_dict = {}
    print("Tuning Results (mean ± std):")
    for k in sum_dict:
        mean = sum_dict[k] / count
        variance = sumsq_dict[k] / count - mean * mean
        std = math.sqrt(variance) if variance > 0 else 0.0
        result_dict[k] = (mean, std)
        print(f"{k}: {mean:.4f} ± {std:.4f}")

    return result_dict


if __name__ == "__main__":
    ARGS = argparser()
    ARGS.PID = os.getpid()
    PrintARGSInfo(ARGS)
    
    torch.manual_seed(12)
    if ARGS.runType == "Pretrain" or ARGS.runType == "FullStage":
        Pretrain(ARGS)

    tuningResultList = []
    for i in range(10):
        torchSeed = i + 3047
        taskAvgPerformance = FineTuneMultiDownstreamTasks(ARGS, torchSeed)
        tuningResultList.append(taskAvgPerformance)
    print("=======================Overall Results==================")
    GetMultiTuningResult(tuningResultList)
