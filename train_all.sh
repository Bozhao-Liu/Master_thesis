#python3 train.py --train True --resume False --network basemodel --loss BCE
#python3 train.py --train True --resume False --network basemodel --loss EXP_BCE
#python3 train.py --train True --resume False --network basemodel --loss focal
#python3 train.py --train True --resume False --network basemodel --loss EXPBCE_Focal_Balanced
#python3 train.py --train True --resume False --network basemodel --loss EXPBCE_BCE_Balanced

python3 train.py --train True --resume False --network alexnet --loss BCE
python3 train.py --train True --resume False --network alexnet --loss EXP_BCE
python3 train.py --train True --resume False --network alexnet --loss focal
python3 train.py --train True --resume False --network alexnet --loss EXPBCE_Focal_Balanced
python3 train.py --train True --resume False --network alexnet --loss EXPBCE_BCE_Balanced

#python3 train.py --train True --resume False --network smallresnet --loss BCE
#python3 train.py --train True --resume False --network smallresnet --loss EXP_BCE
#python3 train.py --train True --resume False --network smallresnet --loss focal
#python3 train.py --train True --resume False --network smallresnet --loss EXPBCE_Focal_Balanced
#python3 train.py --train True --resume False --network smallresnet --loss EXPBCE_BCE_Balanced

#python3 train.py --train True --resume False --network densenet --loss BCE
#python3 train.py --train True --resume False --network densenet --loss EXP_BCE
#python3 train.py --train True --resume False --network densenet --loss focal
#python3 train.py --train True --resume False --network densenet --loss EXPBCE_Focal_Balanced
#python3 train.py --train True --resume False --network densenet --loss EXPBCE_BCE_Balanced
