## 模型训练入口
```bash
conda activate torch-gpu5060

python scripts/train.py -c configs/STGCN_PEMS04_smoke.json
python scripts/train.py -c configs/STGCN_PEMS04.json

python scripts/train.py -c configs/STGCN_PEMS08_smoke.json
python scripts/train.py -c configs/STGCN_PEMS08.json

python scripts/train.py -c configs/STID_PEMS04_smoke.json
python scripts/train.py -c configs/STID_PEMS04.json

python scripts/train.py -c configs/STID_PEMS08_smoke.json
python scripts/train.py -c configs/STID_PEMS08.json
```

## horizon-wise图绘制
```bash
python scripts/horizon_eval.py --model_path outputs/smoke_STID_PEMS04/model_epoch_1.pth
python scripts/horizon_eval.py --model_path outputs/STID_PEMS04/model_epoch_90.pth

python scripts/horizon_eval.py --model_path outputs/smoke_STID_PEMS08/model_epoch_1.pth
python scripts/horizon_eval.py --model_path outputs/STID_PEMS08/model_epoch_96.pth

python scripts/horizon_eval.py --model_path outputs/smoke_STGCN_PEMS04/model_epoch_1.pth
python scripts/horizon_eval.py --model_path outputs/STGCN_PEMS04/model_epoch_12.pth

python scripts/horizon_eval.py --model_path outputs/smoke_STGCN_PEMS08/model_epoch_1.pth