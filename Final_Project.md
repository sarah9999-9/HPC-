# Mini-HPC and Hybrid HPC-Big Data Clusters Project

## Overview

This project involved designing and implementing a traditional High-Performance Computing (HPC) cluster and a hybrid HPC-Big Data cluster using virtual machines. The main goals were to:

- Establish a 3-node cluster (1 master, 2 workers) for distributed machine learning (ML) tasks.
- Perform distributed ML using MPI and mpi4py.
- Create a Docker Swarm-based Spark cluster for scalable big data analytics.
- Execute and assess distributed ML models on standard (MNIST) and bioinformatics gene expression datasets.

## Approach

**Cluster Configuration**

- Set up three Ubuntu 24.04 virtual machines (1 master, 2 workers) using VirtualBox.
- Configured network settings and established passwordless SSH for efficient node communication, validated by accessing workers from the master node.
- Documented setup procedures and resolved issues.

**Task 1: Mini-HPC Cluster with MPI**

- Installed OpenMPI and mpi4py across all nodes.
- Defined a hostfile specifying cluster nodes.
- Executed distributed ML training on the MNIST dataset using a custom Python script (`distributed_mnist.py`), distributing data and computation across nodes.

**Task 2: Hybrid HPC + Big Data Cluster with Spark**

- Initialized Docker Swarm on all three VMs.
- Deployed Apache Spark in cluster mode via a Docker Compose YAML configuration.
- Confirmed Spark cluster functionality and worker registration through the Spark Web UI.
- Conducted distributed ML tasks, such as gene expression analysis, using PySpark on bioinformatics datasets.

## Findings

**Task 1: Distributed MNIST ML (MPI)**

- Training data was evenly distributed across processes
- Accuracy: **96.2%**
- Average Training Time: **0.041 seconds**
- Total Execution Time: **0.69 seconds**

**Task 2: Spark Cluster (PySpark ML)**

- Successfully launched a Spark cluster with 2 worker nodes (2 cores, 2GB RAM each).
- Verified worker registration via the Spark Web UI, ensuring proper configuration.
- Cluster was fully operational for distributed ML on bioinformatics gene expression data.

## Summary

This project provided hands-on experience in:

- Configuring traditional HPC and hybrid Big Data clusters.
- Implementing distributed ML workflows with MPI and Spark.
- Handling real-world bioinformatics data in a distributed, scalable environment.

**Key Insights:**

- Effective network and SSH setup is critical for cluster communication.
- MPI facilitates parallelism for compute-intensive tasks.
- Spark, combined with Docker Swarm, offers robust scalability for big data processing.
- Distributed ML involves challenges like data partitioning and resource management.

This project laid a strong foundation in distributed computing, with practical applications in machine learning and bioinformatics.