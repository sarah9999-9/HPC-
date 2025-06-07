# Import the MPI4Py library for parallel computing using Message Passing Interface (MPI)
from mpi4py import MPI
# Import pandas for data manipulation and analysis
import pandas as pd
# Import NumPy for numerical operations
import numpy as np
# Import RandomForestClassifier from scikit-learn for machine learning
from sklearn.ensemble import RandomForestClassifier

# Initialize MPI communicator for inter-process communication
comm = MPI.COMM_WORLD
# Get the rank (ID) of the current process
rank = comm.Get_rank()
# Get the total number of processes in the communicator
size = comm.Get_size()

# Check if the current process is the root process (rank 0)
if rank == 0:
    # Read the gene expression data from a CSV file, assuming genes as rows and samples as columns
    df = pd.read_csv("large_synthetic_gene_expression.csv")
    
    # Transpose the DataFrame to have samples as rows and genes as columns, set first column as index
    df = df.set_index(df.columns[0]).T.reset_index()
    
    # Extract numerical data (features) from all columns except the first (index) and last (target)
    data = df.iloc[:, 1:-1].values.astype(float)
    # Extract the target variable from the last column
    target = df.iloc[:, -1].values
# If not the root process, initialize data and target as None
else:
    data = None
    target = None

# Broadcast the feature data to all processes from the root process
data = comm.bcast(data, root=0)
# Broadcast the target data to all processes from the root process
target = comm.bcast(target, root=0)

# Calculate the size of the data chunk for each process by dividing total samples by number of processes
chunk_size = len(data) // size
# Calculate the start index for the current process's data chunk
start = rank * chunk_size
# Calculate the end index for the current process's data chunk, handling the last process
end = (rank + 1) * chunk_size if rank != size - 1 else len(data)

# Extract the local subset of feature data for the current process
local_data = data[start:end]
# Extract the local subset of target data for the current process
local_target = target[start:end]

# Initialize a Random Forest Classifier with 100 trees and a fixed random seed for reproducibility
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier on the local data chunk
clf.fit(local_data, local_target)

# Calculate the classification accuracy score on the local data chunk
local_score = clf.score(local_data, local_target)
# Gather the accuracy scores from all processes to the root process
scores = comm.gather(local_score, root=0)

# If the current process is the root process, print the results
if rank == 0:
    # Print the classification scores collected from all processes
    print(f"Classification scores from {size} nodes: {scores}")
    # Calculate and print the average classification accuracy across all processes
    print(f"Average classification accuracy: {np.mean(scores):.4f}")