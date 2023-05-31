
#python -u generate_mnist.py noniid - dir > mnist_dataset.out 2>&1
-data mnist-0.1-npz
nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -nc 20 -nb 10  -m cnn -algo SCAFFOLD -et 1 -li 2 -rp 80 -did 0 >SCAFFOLD_log.out 2>&1 &
nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -nc 20 -nb 10  -m cnn -algo FedALA -et 1 -li 2 -rp 80 -did 0 >log.out 2>&1 &
nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -nc 20 -nb 10  -m cnn -algo ALAJS -et 1 -li 2 -rp 80 -did 0 >log.out 2>&1 &

nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -gr 200 -nc 20 -nb 10  -m cnn -algo SCAFFOLD -et 1 -li 2 -rp 80 -did 0 >SCAFFOLD_log.out 2>&1 &
nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -gr 200 -nc 20 -nb 10  -m cnn -algo FedALA -et 1 -li 2 -rp 80 -did 0 >FedALA_log.out 2>&1 &
nohup python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -gr 200 -nc 20 -nb 10  -m cnn -algo ALAJS -et 1 -li 2 -rp 80 -did 0 >ALAJS_log.out 2>&1 &