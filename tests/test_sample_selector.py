import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from sample_selector import SampleSelector, k_medoids, filter_clean_samples


class TestSampleSelector(unittest.TestCase):

    def setUp(self):
        # Mock the transformer model and tokenizer to avoid actual loading
        self.patcher1 = patch("transformers.AutoTokenizer.from_pretrained")
        self.patcher2 = patch("transformers.AutoModel.from_pretrained")
        self.mock_tokenizer = self.patcher1.start()
        self.mock_model = self.patcher2.start()

        # Configure mocks
        self.mock_tokenizer.return_value = MagicMock()
        self.mock_model.return_value = MagicMock()

        # Create SampleSelector instance with mocked dependencies
        self.selector = SampleSelector(model_name="mock-model", device="cpu")

        # Define some test data
        self.test_texts = [
            "This is a positive sample.",
            "This is a negative example.",
            "Another positive text.",
            "Yet another negative text.",
        ]

        # Mock the embeddings behavior
        self.mock_embeddings = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.15, 0.25, 0.35], [0.45, 0.55, 0.65]]
        )

        # Sample with loss values (as described in the paper)
        self.loss_values = np.array([0.1, 0.9, 0.2, 0.8])
        self.samples_with_loss = list(
            zip(
                self.test_texts,
                ["positive", "negative", "positive", "negative"],
                self.loss_values,
            )
        )

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()

    def test_init(self):
        """Test initialization parameters are correctly set"""
        # Test with default parameters
        selector = SampleSelector()
        self.assertEqual(
            selector.device, "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.assertEqual(selector.max_length, 512)

        # Test with custom parameters
        custom_selector = SampleSelector(
            model_name="custom-model", device="cpu", max_length=256
        )
        self.assertEqual(custom_selector.device, "cpu")
        self.assertEqual(custom_selector.max_length, 256)

    def test_get_embeddings_empty_input(self):
        """Test behavior with empty input"""
        result = self.selector._get_embeddings([])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,))

    def test_get_embeddings_batching(self):
        """Test that batching works correctly for embedding generation"""
        # Create batch outputs with the correct shape
        batch_hidden_state = torch.tensor(
            [
                [[0.1, 0.2, 0.3]],  # First text embedding
                [[0.4, 0.5, 0.6]],  # Second text embedding
            ]
        )

        # Create a mock output for each batch
        batch1_output = MagicMock()
        batch1_output.last_hidden_state = batch_hidden_state

        batch2_output = MagicMock()
        batch2_output.last_hidden_state = batch_hidden_state

        # Make model return a different output for each call
        self.selector.model.side_effect = [batch1_output, batch2_output]

        # Create a mock tokenizer output object with the .to() method
        tokenizer_output = MagicMock()
        tokenizer_output.to.return_value = tokenizer_output
        self.selector.tokenizer.return_value = tokenizer_output

        # Test batching with 2 items per batch (4 texts should need 2 batches)
        result = self.selector._get_embeddings(self.test_texts, batch_size=2)

        # Verify model and tokenizer were called twice (once per batch)
        self.assertEqual(self.selector.tokenizer.call_count, 2)
        self.assertEqual(self.selector.model.call_count, 2)

        # Verify result has the right length
        self.assertEqual(len(result), len(self.test_texts))

        # Verify each item in result has the right shape (embedding dimension)
        self.assertEqual(result.shape[1], 3)  # embedding dimension is 3

    def test_filter_clean_samples_empty_input(self):
        """Test filtering with empty input"""
        result = filter_clean_samples([])
        self.assertEqual(result, [])

    def test_filter_clean_samples_as_per_paper(self):
        """
        Test filtering clean samples following the paper's methodology.

        The paper uses GMM to model clean vs. noisy samples based on loss values.
        """
        # Create samples with loss values where smaller losses are expected to be clean
        with patch("sklearn.mixture.GaussianMixture") as mock_gmm:
            mock_gmm_instance = MagicMock()
            mock_gmm.return_value = mock_gmm_instance
            mock_gmm_instance.means_ = np.array(
                [[0.15], [0.85]]
            )  # Clean vs noisy means

            # Set probabilities to clearly separate clean/noisy samples
            mock_gmm_instance.predict_proba.return_value = np.array(
                [
                    [0.9, 0.1],  # Sample 0: high probability of clean
                    [0.1, 0.9],  # Sample 1: high probability of noisy
                    [0.8, 0.2],  # Sample 2: high probability of clean
                    [0.2, 0.8],  # Sample 3: high probability of noisy
                ]
            )

            clean_samples = filter_clean_samples(self.samples_with_loss)

            # According to the paper, samples with smaller loss should be considered clean
            self.assertEqual(len(clean_samples), 2)

            # Check if the right samples are selected (the ones with lower loss)
            clean_texts = [sample[0] for sample in clean_samples]
            self.assertIn("This is a positive sample.", clean_texts)
            self.assertIn("Another positive text.", clean_texts)

    @patch("sklearn.mixture.GaussianMixture")
    def test_gmm_component_selection(self, mock_gmm):
        """
        Test that the GMM selects the component with the smallest mean as clean.
        This is a key aspect of the paper's approach.
        """
        # Configure GMM mock with two distinct components
        mock_gmm_instance = MagicMock()
        mock_gmm.return_value = mock_gmm_instance

        # First component has smaller mean (clean) and second is larger (noisy)
        mock_gmm_instance.means_ = np.array([[0.15], [0.85]])

        # Set probabilities to clearly separate samples
        mock_gmm_instance.predict_proba.return_value = np.array(
            [
                [0.9, 0.1],  # High probability of clean component
                [0.1, 0.9],  # High probability of noisy component
                [0.8, 0.2],  # High probability of clean component
                [0.2, 0.8],  # High probability of noisy component
            ]
        )

        # Test filtering
        clean_samples = filter_clean_samples(self.samples_with_loss)

        # Should select samples 0 and 2 which have high probability of clean component
        self.assertEqual(len(clean_samples), 2)
        clean_texts = [sample[0] for sample in clean_samples]
        self.assertIn(self.test_texts[0], clean_texts)
        self.assertIn(self.test_texts[2], clean_texts)

    def test_k_medoids_empty_input(self):
        """Test k-medoids with empty input"""
        result = k_medoids(np.array([]), 2)
        self.assertEqual(result, [])

    def test_k_medoids_fewer_samples_than_k(self):
        """Test k-medoids when there are fewer samples than k"""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = k_medoids(embeddings, 5)
        self.assertEqual(set(result), {0, 1})  # Should return all indices

    def test_k_medoids_clustering(self):
        """
        Test k-medoids clustering following the paper's methodology.

        The paper uses k-medoids to find representative samples from each class.
        """
        # Define embeddings with clear clusters
        embeddings = np.array(
            [
                [0, 0],  # Cluster 1
                [0.1, 0.1],  # Cluster 1
                [5, 5],  # Cluster 2
                [5.1, 5.1],  # Cluster 2
            ]
        )

        # Mock KMeans to make the test behavior deterministic
        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans.return_value = mock_kmeans_instance

            # Set up KMeans to define two clear clusters
            mock_kmeans_instance.fit.return_value = mock_kmeans_instance
            mock_kmeans_instance.cluster_centers_ = np.array(
                [
                    [0.05, 0.05],  # Center for cluster 1
                    [5.05, 5.05],  # Center for cluster 2
                ]
            )

            # Find 2 medoids
            with patch("sklearn.metrics.pairwise_distances") as mock_distances:
                # First call: distances from points to centers
                mock_distances.side_effect = [
                    # Points to centers distances:
                    np.array(
                        [
                            [
                                0.07,
                                7.14,
                            ],  # Sample 0: close to center 1, far from center 2
                            [
                                0.07,
                                7.00,
                            ],  # Sample 1: close to center 1, far from center 2
                            [
                                7.07,
                                0.07,
                            ],  # Sample 2: far from center 1, close to center 2
                            [
                                7.14,
                                0.07,
                            ],  # Sample 3: far from center 1, close to center 2
                        ]
                    ),
                    # Within-cluster distances for cluster 1 (samples 0, 1)
                    np.array([[0.07], [0.07]]),
                    # Within-cluster distances for cluster 2 (samples 2, 3)
                    np.array([[0.07], [0.07]]),
                ]

                # This should select one medoid from each cluster
                medoid_indices = k_medoids(embeddings, 2)

                # Should have 2 medoids
                self.assertEqual(len(medoid_indices), 2)

                # One should be from cluster 1 (0 or 1) and one from cluster 2 (2 or 3)
                self.assertTrue(0 in medoid_indices or 1 in medoid_indices)
                self.assertTrue(2 in medoid_indices or 3 in medoid_indices)

    @patch.object(SampleSelector, "_get_embeddings")
    def test_select_demonstrations_class_balance(self, mock_get_embeddings):
        """
        Test that demonstration selection maintains class balance as per the paper.

        The paper selects demonstrations per class to ensure balanced representation.
        """
        # Set up mock embeddings for 4 samples (2 per class)
        mock_get_embeddings.return_value = np.array(
            [
                [0, 0],  # Class A, Sample 1
                [0.1, 0.1],  # Class A, Sample 2
                [5, 5],  # Class B, Sample 1
                [5.1, 5.1],  # Class B, Sample 2
            ]
        )

        # Create balanced class samples
        balanced_samples = [
            ("Sample A1", "class_a"),
            ("Sample A2", "class_a"),
            ("Sample B1", "class_b"),
            ("Sample B2", "class_b"),
        ]

        # Mock k_medoids to make test deterministic
        with patch("sample_selector.k_medoids") as mock_k_medoids:
            # Return first sample from each class
            mock_k_medoids.side_effect = [
                [0],  # Class A: select first sample
                [0],  # Class B: select first sample
            ]

            # Request 1 sample per class
            demonstrations = self.selector.select_demonstrations(
                balanced_samples, num_per_class=1
            )

            # Should get exactly 2 demonstrations (1 per class)
            self.assertEqual(len(demonstrations), 2)

            # Check class balance
            demo_classes = [demo[1] for demo in demonstrations]
            self.assertEqual(demo_classes.count("class_a"), 1)
            self.assertEqual(demo_classes.count("class_b"), 1)

    def test_select_demonstrations_small_dataset(self):
        """Test demonstration selection with fewer samples than requested per class"""
        # Small dataset with 1 sample per class
        small_samples = [
            ("Only class A sample", "class_a"),
            ("Only class B sample", "class_b"),
        ]

        # Request more demos than available
        demos = self.selector.select_demonstrations(small_samples, num_per_class=5)

        # Should get all available samples, maintaining class balance
        self.assertEqual(len(demos), 2)
        demo_classes = [demo[1] for demo in demos]
        self.assertEqual(demo_classes.count("class_a"), 1)
        self.assertEqual(demo_classes.count("class_b"), 1)

    def test_combined_filtering_and_selection(self):
        """
        Test the combined method that performs filtering and demonstration selection.
        This tests the entire pipeline as described in the paper.
        """
        # Create samples with varying loss values across classes
        samples_with_loss = [
            ("Clean A1", "class_a", 0.1),  # Clean
            ("Noisy A2", "class_a", 0.9),  # Noisy
            ("Clean A3", "class_a", 0.2),  # Clean
            ("Clean B1", "class_b", 0.15),  # Clean
            ("Noisy B2", "class_b", 0.85),  # Noisy
            ("Clean B3", "class_b", 0.25),  # Clean
        ]

        # Mock the entire pipeline to focus on testing the integration
        with patch("sample_selector.filter_clean_samples") as mock_filter:
            # Return a filtered set of clean samples
            mock_filter.return_value = [
                ("Clean A1", "class_a"),
                ("Clean A3", "class_a"),
                ("Clean B1", "class_b"),
                ("Clean B3", "class_b"),
            ]

            with patch.object(SampleSelector, "select_demonstrations") as mock_select:
                # Return balanced demonstrations
                mock_select.return_value = [
                    ("Clean A1", "class_a"),
                    ("Clean B1", "class_b"),
                ]

                # Run the combined method
                demonstrations = self.selector.select_demonstrations_from_noisy(
                    samples_with_loss, num_per_class=1
                )

                # Should get 2 demonstrations (1 per class)
                self.assertEqual(len(demonstrations), 2)

                # Verify the filtering step was called with the right arguments
                mock_filter.assert_called_once_with(samples_with_loss, 0.7)

                # Verify the selection step was called with the filtered samples
                mock_select.assert_called_once_with(mock_filter.return_value, 1)


if __name__ == "__main__":
    unittest.main()
