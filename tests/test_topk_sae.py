"""
Unit tests for TopKSAE module.
"""

import pytest
import torch
import torch.nn.functional as F

from agora_sae.model.topk_sae import TopKSAE, TopKSAEWithResampling


class TestTopKSAE:
    """Test suite for TopKSAE."""
    
    @pytest.fixture
    def sae(self):
        """Create a small SAE for testing."""
        return TopKSAE(d_model=64, d_sae=256, k=8)
    
    @pytest.fixture
    def batch(self):
        """Create a random batch of activations."""
        return torch.randn(32, 64)  # [batch_size, d_model]
    
    def test_init(self, sae):
        """Test initialization."""
        assert sae.d_model == 64
        assert sae.d_sae == 256
        assert sae.k == 8
        
        # Check shapes
        assert sae.W_enc.shape == (64, 256)
        assert sae.W_dec.shape == (256, 64)
        assert sae.b_enc.shape == (256,)
        assert sae.b_dec.shape == (64,)
        
    def test_decoder_unit_norm(self, sae):
        """Test that decoder columns have unit norm after initialization."""
        # W_dec is [d_sae, d_model], each row should have unit norm
        norms = torch.norm(sae.W_dec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
    def test_set_decoder_norm(self, sae):
        """Test set_decoder_norm constraint."""
        # Corrupt the norms
        sae.W_dec.data *= 2.0
        
        # Apply constraint
        sae.set_decoder_norm()
        
        # Check norms are back to 1
        norms = torch.norm(sae.W_dec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
    def test_forward_shapes(self, sae, batch):
        """Test forward pass output shapes."""
        x_hat, f, topk_indices, z = sae(batch)
        
        assert x_hat.shape == batch.shape
        assert f.shape == (32, 256)  # [batch_size, d_sae]
        assert topk_indices.shape == (32, 8)  # [batch_size, k]
        assert z.shape == (32, 256)  # [batch_size, d_sae]
        
    def test_topk_sparsity(self, sae, batch):
        """Test that exactly k features are active per sample."""
        _, f, _, _ = sae(batch)
        
        # Count non-zero elements per sample
        active_counts = (f > 0).sum(dim=1)
        
        # Should be exactly k for each sample
        assert torch.all(active_counts == sae.k)
        
    def test_compute_loss(self, sae, batch):
        """Test loss computation."""
        x_hat, f, topk_indices, z = sae(batch)
        loss_dict = sae.compute_loss(batch, x_hat, f, topk_indices, z)
        
        assert "loss" in loss_dict
        assert "l_recon" in loss_dict
        assert "l_aux" in loss_dict
        assert "l2_ratio" in loss_dict
        assert "l0" in loss_dict
        
        # All losses should be non-negative
        assert loss_dict["loss"] >= 0
        assert loss_dict["l_recon"] >= 0
        assert loss_dict["l_aux"] >= 0
        
        # L0 should be approximately k
        assert abs(loss_dict["l0"] - sae.k) < 0.1
        
    def test_encode_decode(self, sae, batch):
        """Test encode and decode separately."""
        z = sae.encode(batch)
        f, indices = sae.activate(z)
        x_hat = sae.decode(f)
        
        # Compare with forward pass
        x_hat_fwd, f_fwd, _, z_fwd = sae(batch)
        
        assert torch.allclose(z, z_fwd)
        assert torch.allclose(f, f_fwd)
        assert torch.allclose(x_hat, x_hat_fwd)
        
    def test_gradient_flow(self, sae, batch):
        """Test that gradients flow correctly."""
        x_hat, f, topk_indices, z = sae(batch)
        loss_dict = sae.compute_loss(batch, x_hat, f, topk_indices, z)
        
        loss_dict["loss"].backward()
        
        # Check gradients exist
        assert sae.W_enc.grad is not None
        assert sae.W_dec.grad is not None
        assert sae.b_enc.grad is not None
        assert sae.b_dec.grad is not None
        
        # Gradients should not be all zero
        assert sae.W_enc.grad.abs().sum() > 0
        assert sae.W_dec.grad.abs().sum() > 0
        
    def test_activation_stats(self, sae, batch):
        """Test activation statistics tracking."""
        # Initially all zeros
        assert torch.all(sae.latent_activation_count == 0)
        
        # Forward and update stats
        _, _, topk_indices, _ = sae(batch)
        sae.update_activation_stats(topk_indices)
        
        # Some latents should now have counts
        assert sae.latent_activation_count.sum() > 0
        
        # Steps since activation should be 0 for active latents
        active_latents = topk_indices.unique()
        assert torch.all(sae.steps_since_activation[active_latents] == 0)
        
    def test_dead_latent_detection(self, sae):
        """Test dead latent detection."""
        # Simulate many steps without activation for some latents
        sae.steps_since_activation[:100] = 20000  # More than threshold
        
        dead_ratio = sae.get_dead_latent_ratio()
        assert dead_ratio > 0
        
        dead_indices = sae.get_dead_latent_indices()
        assert len(dead_indices) == 100


class TestTopKSAEWithResampling:
    """Test suite for TopKSAE with resampling."""
    
    @pytest.fixture
    def sae(self):
        """Create a small SAE with resampling for testing."""
        return TopKSAEWithResampling(d_model=64, d_sae=256, k=8)
    
    @pytest.fixture
    def batch(self):
        """Create a random batch of activations."""
        return torch.randn(32, 64)
    
    def test_resample_dead_latents(self, sae, batch):
        """Test dead latent resampling."""
        # Mark some latents as dead
        sae.steps_since_activation[:50] = 20000
        
        # Forward pass
        x_hat, _, _, _ = sae(batch)
        
        # Resample
        n_resampled = sae.resample_dead_latents(batch, x_hat)
        
        assert n_resampled == 50
        
        # Dead latents should now have reset counters
        assert torch.all(sae.steps_since_activation[:50] == 0)
        
        # Decoder should still have unit norms after resampling
        norms = torch.norm(sae.W_dec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestSAEReconstruction:
    """Test reconstruction quality."""
    
    def test_reconstruction_improves_with_training(self):
        """Test that reconstruction error decreases with training."""
        sae = TopKSAE(d_model=64, d_sae=256, k=8)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        # Use a fixed batch for testing
        batch = torch.randn(128, 64)
        
        # Initial loss
        x_hat, f, topk_indices, z = sae(batch)
        initial_loss = sae.compute_loss(batch, x_hat, f, topk_indices, z)["l_recon"].item()
        
        # Train for a few steps
        for _ in range(100):
            optimizer.zero_grad()
            x_hat, f, topk_indices, z = sae(batch)
            loss_dict = sae.compute_loss(batch, x_hat, f, topk_indices, z)
            loss_dict["loss"].backward()
            optimizer.step()
            sae.set_decoder_norm()
            
        # Final loss should be lower
        x_hat, f, topk_indices, z = sae(batch)
        final_loss = sae.compute_loss(batch, x_hat, f, topk_indices, z)["l_recon"].item()
        
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
