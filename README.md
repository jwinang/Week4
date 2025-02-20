# Week4 Unsupervised Learning

This repository contains code and explanations for various unsupervised learning methods, focusing on applications in Earth Observation (EO) data analysis. This repository provides practical guidance on applying unsupervised learning techniques, such as **K-means clustering** and **Gaussian Mixture Models (GMM)**, for classification tasks in environmental data analysis.

---

## Installation

### **Requirements**

It will be running in Google Colab, mount Google Drive and install additional dependencies:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install rasterio
!pip install netCDF4
```

---

## Unsupervised Learning Algorithms

### **1. K-means Clustering**
K-means clustering partitions a dataset into *k* groups (clusters) by minimizing intra-cluster variance. The algorithm follows these steps:

1. Initialize *k* cluster centroids.
2. Assign each data point to the nearest centroid.
3. Update centroids based on the mean of assigned points.
4. Repeat until centroids converge.

#### **Why Use K-means?**
- Efficient and scalable for large datasets.
- Easy to interpret results.
- Suitable for exploratory data analysis.

```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

![image](https://github.com/user-attachments/assets/c0a4faff-8a9b-489d-8119-8d689eea5a52)

```

---

### **2. Gaussian Mixture Models (GMM)**
Gaussian Mixture Models (GMM) assume that data is generated from a mixture of Gaussian distributions. It employs the Expectation-Maximization (EM) algorithm to estimate parameters.

#### **Why Use GMM?**
- Supports soft clustering (probabilistic assignment of points to clusters).
- More flexible than K-means in handling cluster shapes.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()

![image](https://github.com/user-attachments/assets/b0c256c4-3cd2-49ff-a4fe-5ad539924636)

```

---

## Image Classification
Now, let's explore the application of these unsupervised methods to image classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-2 imagery.

## K-Means Implementation

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
![image](https://github.com/user-attachments/assets/ffa98617-eeb3-43ee-99a6-e90e5c6500b8)

---

## Running the Notebook
To execute the notebook:
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Chapter1_Unsupervised_Learning_Methods_2.ipynb
   ```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

