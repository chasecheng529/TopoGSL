import argparse
import os
import torch


def argparser():
  parser = argparse.ArgumentParser()
  # ==========================For public use=================================
  parser.add_argument(
    "--DownstreamTasks",
    nargs = '+',
    type = str,
    default = ["BBBP", "TOX21", "ClinTox", "HIV", "BACE", "SIDER", "ESOL", "FreeSolv", "Lipo", "QM7", "QM8"]
  )
  parser.add_argument(
    "--FeatureNums",
    type = int,
    default = 512,
  )
  parser.add_argument(
    "--numWorks",
    type = int,
    default = 8
  )
  parser.add_argument(
      '--GPU',
      type = int,
      default = 2
  )
  parser.add_argument(
    "--runType",
    type = str,
    default = "FullStage",
    help = "Pretrain mode (Pretrain), fine tuning mode (Tuning) and both (FullStage)"
  )
  parser.add_argument(
    "--distributionPretrain",
    type = bool,
    default = False,
  )
  parser.add_argument(
    "--gitNode",
    type = str,
    default = "None",
  )

  # ==================For pretrain use=====================
  parser.add_argument(
    '--pretrainEpoch',
    type = int,
    default = 100
  )
  parser.add_argument(
    '--pretrainWarmUp',
    type = int,
    default = 0
  )
  parser.add_argument(
    "--pretrainBatchSize",
    type = int,
    default = 128
  )
  parser.add_argument(
    "--pretrainConfig",
    nargs = '+',
    type = float,
    default = [5e-4, 1e-5],
    help = "[lr, weight_decay]"
  )
  
  # ====================For fine tuning use==================
  parser.add_argument(
    "--checkPointName",
    type = str,
    default = "f64136",
  )
  parser.add_argument(
    "--TuneBatchSize",
    type = int,
    default = 32
  )
  parser.add_argument(
    '--TuningEpoch',
    type = int,
    default = 100
  )
  parser.add_argument(
    "--tuningConfig",
    nargs = '+',
    type = float,
    default = [5e-4, 1e-4, 1e-5],
    help = "[prehead_lr, base_lr, weight_decay]"
  )

    # ==================For activity cliff use=====================
  parser.add_argument(
      '--NameIndex',
      type = int,
      default = 1
    )


  args, unparsed = parser.parse_known_args()
  args.device = torch.device("cuda:" + str(args.GPU))
  return args