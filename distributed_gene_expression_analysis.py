# Import SparkSession from PySpark for creating a Spark session
from pyspark.sql import SparkSession
# Import RandomForestClassifier from PySpark's ML library for machine learning
from pyspark.ml.classification import RandomForestClassifier
# Import VectorAssembler from PySpark's ML library to combine features into a single vector
from pyspark.ml.feature import VectorAssembler
# Import pandas for potential data manipulation (not used in this script)
import pandas as pd

# Create or retrieve a SparkSession with the application name "BioinformaticsML"
spark = SparkSession.builder.appName("BioinformaticsML").getOrCreate()

# Read the gene expression data from a CSV file, enabling header and schema inference
df = spark.read.csv("large_synthetic_gene_expression.csv", header=True, inferSchema=True)
# Get all column names except the last one to use as feature columns
feature_cols = df.columns[:-1]
# Create a VectorAssembler to combine feature columns into a single "features" column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# Transform the DataFrame to include the assembled features column
data = assembler.transform(df)
# Split the data into training (80%) and test (20%) sets randomly
train_data, test_data = data.randomSplit([0.8, 0.2])

# Initialize a Random Forest Classifier with the last column as the label and 100 trees
rf = RandomForestClassifier(labelCol=df.columns[-1], featuresCol="features", numTrees=100)
# Train the Random Forest model on the training data
model = rf.fit(train_data)

# Generate predictions on the test data using the trained model
predictions = model.transform(test_data)
# Calculate accuracy by comparing the label column with predictions and dividing by total test samples
accuracy = predictions.filter(predictions[df.columns[-1]] == predictions.prediction).count() / test_data.count()

# Print the accuracy of the model on the test data
print("Accuracy on test data:", accuracy)

# Stop the SparkSession to release resources
spark.stop()