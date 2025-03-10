## Gaussian Mixture Model (GMM) - What is it?

A Gaussian Mixture Model is a statistical model that assumes data points come from a mixture of several Gaussian distributions (bell curves). In the FreeAL context, it's used to separate "clean" from "noisy" samples.
Think of it this way: When you plot the loss values for all your samples, you might see something like two overlapping bell curves - one with lower loss values (clean samples) and another with higher loss values (noisy samples).
GMM - How it works in the code:

The code calculates the loss for each sample (how confidently the model predicts its label)
It then fits a GMM with 2 components to these loss values:
```python
# From sample_selector.py
losses = np.array([sample[2] for sample in samples_with_loss]).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(losses)
```

It identifies which component has the smaller mean (the "clean" component)
Each sample gets a probability of belonging to the clean component
Samples with probability above a threshold (0.7) are considered clean:
```python
clean_samples = [
    (sample[0], sample[1])
    for i, sample in enumerate(samples_with_loss)
    if probs[i] >= threshold
]
```


k-medoids - What is it?
k-medoids is a clustering algorithm similar to k-means, but instead of using computed centroids, it uses actual data points (medoids) as the centers of clusters. It's better than k-means when you want to actually select representative examples.
Think of it like this: If you have a bunch of points in space, k-medoids helps you pick k points that best represent different regions of that space.
k-medoids - How it works in the code:

The SLM generates embeddings (vector representations) for each clean text sample
The k-medoids algorithm selects representative samples from each class:
```python
# From sample_selector.py
embeddings = self._get_embeddings(texts)
```

## Use k-medoids to select diverse examples
```python
medoid_indices = k_medoids(embeddings, num_per_class)
```


It does this by:

First using KMeans to find initial centers
Assigning each point to the nearest center
Finding the actual data point closest to each center
Using these data points as medoids (representatives)

```python
# Find the point with minimum distance to center
within_distances = pairwise_distances(cluster_points, center)
medoid_idx = cluster_points_indices[np.argmin(within_distances)]
```

These two techniques work together to:

First separate clean from noisy samples (GMM)
Then select diverse, representative examples from the clean samples (k-medoids)

This ensures that in each round, the LLM gets high-quality but diverse examples to learn from, preventing the narrowing problem you were concerned about.
All of this is implemented in the sample_selector.py file, particularly in the `filter_clean_samples` and `k_medoids` functions and the `select_demonstrations` method.