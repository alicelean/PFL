
#python -u generate_mnist.py noniid - dir > mnist_dataset.out 2>&1
python -u /Users/alice/Desktop/python/PFL/system/main.py -t 1 -jr 0.1 -nc 20 -nb 10 -data mnist-0.1-npz -m cnn -algo FedALA -et 1 -li 2 -rp 80 -did 0 >log.out 2>&1 &