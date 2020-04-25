python3 train.py --train True --network alexnet --loss BCE --lrDecay 0.9
python3 train.py --train True --network alexnet --loss EXP_BCE --lrDecay 0.9
python3 train.py --train True --network alexnet --loss focal --lrDecay 0.9
python3 train.py --train True --network alexnet --loss EXPBCE_Focal_Balanced --lrDecay 0.9
python3 train.py --train True --network alexnet --loss EXPBCE_BCE_Balanced --lrDecay 0.9

python3 train.py --train True --network alexnet --loss BCE --lrDecay 0.8
python3 train.py --train True --network alexnet --loss EXP_BCE --lrDecay 0.8
python3 train.py --train True --network alexnet --loss focal --lrDecay 0.8
python3 train.py --train True --network alexnet --loss EXPBCE_Focal_Balanced --lrDecay 0.8
python3 train.py --train True --network alexnet --loss EXPBCE_BCE_Balanced --lrDecay 0.8

