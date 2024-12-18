from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from numba import jit
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# set board size
board_size = 6

# read game data
data = pd.read_csv('player0_6.csv', header=None, nrows=100000)
data2 = pd.read_csv('player1_6.csv', header=None, nrows=100000)

arrays = []
p0w = 0
p1w = 0

# Function to reshape and validate each row
def process_row(row):
    line_data = np.array(list(row[0]), dtype=str)
    if line_data.size != board_size * board_size:
        print(f"Error: Row size {line_data.size} is not {board_size * board_size}.")
        return None
    return line_data.reshape(board_size, board_size)

# Process data for player 0
for index, row in data.iterrows():
    line_matrix = process_row(row)
    if line_matrix is not None:
        arrays.append(line_matrix)
        p0w += 1

# Process data for player 1
for index, row in data2.iterrows():
    line_matrix = process_row(row)
    if line_matrix is not None:
        arrays.append(line_matrix)
        p1w += 1
        
# store data into a 3D array   
array_3d = np.array(arrays)    
print(p0w,p1w)
result = np.empty(p0w+p1w, dtype=np.uint32)
for i in range(0,p0w):
    result[i] = 0
for i in range(p0w,p0w+p1w):
    result[i] = 1

# 70% training and 30% testing data
X_train, X_test, Y_train, Y_test = train_test_split(array_3d, result, test_size=0.3, random_state=42)

# graph tsetlin machine parameters
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=12000, type=int)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=64, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=64, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()


number_of_nodes = board_size * board_size 
symbol_names = []

# neighbor functions
hex_neighbors = [
    (-1,  0), (-1, 1),
    (0,  -1), (0,  1),
    (1, -1), (1,  0)
]

def get_neighbors(i, j, board_size):
    neighbors = []
    for ni, nj in hex_neighbors:
        mi, mj = i + ni, j + nj
        if 0 <= mi < board_size and 0 <= mj < board_size:
            neighbors.append((mi, mj))
    return neighbors

#add symbol names
for i in range(board_size):
    for j in range(board_size):
        symbol_names.append(f"{i}_{j}")
for val in ["0","1","2"]:
    symbol_names.append(val)

#define the type of connetion between cells
def get_edge_property(i, j, ni, nj): 
    if ni == i - 1 and nj == j:  
        return 0
    elif ni == i - 1 and nj == j + 1:
        return 1
    elif ni == i and nj == j - 1:
        return 2
    elif ni == i and nj == j + 1:
        return 3
    elif ni == i + 1 and nj == j - 1:
        return 4
    elif ni == i + 1 and nj == j:
        return 5        

# generate training Graphs
graphs_train = Graphs(X_train.shape[0], symbols=symbol_names, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

# generate nodes
for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

# add edge numbers
for graph_id in range(X_train.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}"
            neighbors = get_neighbors(i, j, board_size)
            graphs_train.add_graph_node(graph_id, node_id, len(neighbors))
graphs_train.prepare_edge_configuration()

# add egde types
for graph_id in range(X_train.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}"  
            neighbors = get_neighbors(i, j, board_size) 
            for ni, nj in neighbors:
                neighbor_id = f"{ni}_{nj}" 
                edge_type = get_edge_property(i, j, ni, nj)
                graphs_train.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)

# add node names
for graph_id in range(X_train.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}"  
            cell_value = X_train[graph_id][i][j]  
            graphs_train.add_graph_node_property(graph_id, node_id, node_id)
            graphs_train.add_graph_node_property(graph_id, node_id, str(cell_value))

graphs_train.encode()
print("Training data prepared")

# generate testing Graphs
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

# generate nodes
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)
graphs_test.prepare_node_configuration()

# add edge numbers
for graph_id in range(X_test.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}"
            neighbors = get_neighbors(i, j, board_size)
            graphs_test.add_graph_node(graph_id, node_id, len(neighbors))
graphs_test.prepare_edge_configuration()

# add edge types
for graph_id in range(X_test.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}" 
            neighbors = get_neighbors(i, j, board_size)  
            for ni, nj in neighbors:
                neighbor_id = f"{ni}_{nj}"  
                edge_type = get_edge_property(i, j, ni, nj)
                graphs_test.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)

# add node names
for graph_id in range(X_test.shape[0]):
    for i in range(board_size):
        for j in range(board_size):
            node_id = f"{i}_{j}"  
            cell_value = X_test[graph_id][i][j]  
            graphs_test.add_graph_node_property(graph_id, node_id, node_id)
            graphs_test.add_graph_node_property(graph_id, node_id, str(cell_value))

graphs_test.encode()

print("Testing data prepared")

# start training
data=[]
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()
    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()
    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
    data.append((i,result_train)) 

weights = tm.get_state()[1].reshape(2, -1)
print(data)

