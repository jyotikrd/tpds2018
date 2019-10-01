Read Me 

In dataset: 1 and -1 label. Replace all non 1 labels to -1:
Replace !1 in column $1 with -1 in unix: 
awk '$1!=1 {$1=-1} {print}' file > newfile

//Shuffle 1000 lines in file in unix: 
shuf -n 1000 newfile > dataset

//Randomize the order of samples in the csv file before splitting
//python randomizer.py <file.xxx>

Convert to CSV: 
python convert2csv.py <sample size> <#attributes> dataset <column index of first whitespace> <class label representing postive>

Run MEKA code on the csv generated file to get new csv file dataX.csv


Split MEKA output file to Train and Test files
./train_test.sh dataX.csv <n> <d> <fraction assigned to train>


// From here refer to any .job file to understand below
Install numpy:
module load numpy/1.10.4-intel-2016a-Python-2.7.11

Instal mpiicpc:
module load impi/5.0.3.048-iccifort-2015.3.187

Install armadillo:
module load Armadillo/6.400.3-intel-2015B-Python-2.7.10


Compile:
mpiicpc -o MPIexecV1 dist_MPI_QRSVM_MEKA_V1.cpp -O3 -larmadillo -std=c++14

Distribute MEKA Training data dataX.csv having <nTrain> samples to <nodes> units, k: new rank from MEKA:
./distribute.sh dataX.csv <nTrain> <k> <nodes>

Run:
mpirun -np <nodes> ./MPIexecV1 <C>  <learnRate>  <thresh>  <gamma>
