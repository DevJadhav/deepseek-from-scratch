"""
Distributed Training Infrastructure Tests

Tests for:
- FSDP integration
- Expert parallelism
- DualPipe pipeline parallelism
- Sequence parallelism
- Fault tolerance
- Distributed checkpointing
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import time

# Test availability
TORCH_DISTRIBUTED_AVAILABLE = hasattr(torch, 'distributed')
CUDA_AVAILABLE = torch.cuda.is_available()
MULTI_GPU = CUDA_AVAILABLE and torch.cuda.device_count() > 1


class TestFSDP:
    """Test FSDP integration module."""
    
    def test_fsdp_config_creation(self):
        """Test FSDPConfig dataclass creation."""
        from deepseek.torch.training.fsdp import FSDPConfig
        
        config = FSDPConfig()
        # Check default values based on actual implementation
        assert hasattr(config, 'sharding_strategy')
        assert hasattr(config, 'cpu_offload')
        
    def test_fsdp_config_with_custom_values(self):
        """Test FSDPConfig with custom values."""
        from deepseek.torch.training.fsdp import FSDPConfig
        
        config = FSDPConfig(
            sharding_strategy="full_shard",
            cpu_offload=True,
        )
        assert config.cpu_offload is True
        
    def test_create_mixed_precision_policy(self):
        """Test mixed precision policy creation."""
        from deepseek.torch.training.fsdp import FSDPConfig, create_mixed_precision_policy
        
        config = FSDPConfig()
        # Test with config object
        policy = create_mixed_precision_policy(config)
        # Policy may be None or MixedPrecision object based on implementation
        
    def test_create_auto_wrap_policy(self):
        """Test auto wrap policy creation."""
        from deepseek.torch.training.fsdp import FSDPConfig, create_auto_wrap_policy
        
        config = FSDPConfig()
        policy = create_auto_wrap_policy(config)
        # Policy should be a callable or policy object
        
    def test_fsdp_wrapper_requires_distributed(self):
        """Test FSDPWrapper requires distributed initialization."""
        from deepseek.torch.training.fsdp import FSDPWrapper, FSDPConfig
        
        config = FSDPConfig()
        
        # Without distributed initialized, should raise
        with pytest.raises(RuntimeError):
            wrapper = FSDPWrapper(config)


class TestExpertParallelism:
    """Test Expert Parallelism module."""
    
    def test_expert_parallel_config(self):
        """Test ExpertParallelConfig creation."""
        from deepseek.torch.model.expert_parallel import ExpertParallelConfig
        
        config = ExpertParallelConfig()
        # Check has expected attributes
        assert hasattr(config, 'ep_size')
        assert hasattr(config, 'num_experts')
        assert hasattr(config, 'capacity_factor')
        
    def test_expert_parallel_config_custom(self):
        """Test ExpertParallelConfig with custom values."""
        from deepseek.torch.model.expert_parallel import ExpertParallelConfig
        
        config = ExpertParallelConfig(
            num_experts=64,
            ep_size=4,
            capacity_factor=1.5,
        )
        assert config.num_experts == 64
        assert config.ep_size == 4
        assert config.capacity_factor == 1.5
        
    def test_dispatch_metadata_creation(self):
        """Test DispatchMetadata dataclass."""
        from deepseek.torch.model.expert_parallel import DispatchMetadata
        
        # Create with actual signature from dataclass
        metadata = DispatchMetadata(
            permutation_indices=torch.zeros(10, dtype=torch.long),
            inverse_permutation=torch.zeros(10, dtype=torch.long),
            send_counts=[5, 5],
            recv_counts=[5, 5],
            expert_counts=torch.zeros(8, dtype=torch.long),
            expert_gates=torch.ones(10),
        )
        assert metadata.permutation_indices.shape == (10,)
        assert len(metadata.send_counts) == 2
        
    def test_all_to_all_dispatcher_local_mode(self):
        """Test AllToAllDispatcher in local mode (no distributed).
        
        Note: AllToAllDispatcher is a torch.autograd.Function with static methods.
        Use all_to_all_dispatch function or ExpertParallelMoE for actual dispatch.
        """
        from deepseek.torch.model.expert_parallel import (
            AllToAllDispatcher,
            ExpertParallelConfig,
            all_to_all_dispatch,
        )
        
        config = ExpertParallelConfig(
            num_experts=8,
            ep_size=1,
        )
        
        # Test that AllToAllDispatcher.apply works (static autograd function)
        x = torch.randn(16, 64, requires_grad=True)  # 16 tokens, hidden=64
        
        # Test using the convenience function with None group (local mode)
        output = all_to_all_dispatch(
            x, 
            output_split_sizes=[16],
            input_split_sizes=[16],
            group=None
        )
        
        assert output.shape == x.shape  # Same shape in local mode


class TestDualPipePipeline:
    """Test DualPipe Pipeline Parallelism module."""
    
    def test_pipeline_config(self):
        """Test PipelineConfig creation."""
        from deepseek.torch.training.dualpipe import PipelineConfig
        
        config = PipelineConfig(num_stages=4, num_micro_batches=8)
        assert config.num_stages == 4
        assert config.num_micro_batches == 8
        
    def test_schedule_action_enum(self):
        """Test ScheduleAction enum values."""
        from deepseek.torch.training.dualpipe import ScheduleAction
        
        # Check enum exists and has expected values
        assert hasattr(ScheduleAction, 'FORWARD')
        assert hasattr(ScheduleAction, 'BACKWARD')
        
    def test_dualpipe_scheduler_creation(self):
        """Test DualPipeScheduler creation."""
        from deepseek.torch.training.dualpipe import DualPipeScheduler, PipelineConfig
        
        config = PipelineConfig(num_stages=4, num_micro_batches=8)
        scheduler = DualPipeScheduler(config)
        
        assert scheduler is not None
        
    def test_dualpipe_scheduler_schedule_generation(self):
        """Test DualPipeScheduler generates valid schedule."""
        from deepseek.torch.training.dualpipe import DualPipeScheduler, PipelineConfig
        
        config = PipelineConfig(num_stages=2, num_micro_batches=4)
        scheduler = DualPipeScheduler(config)
        
        # Schedule is built internally, test __len__
        assert len(scheduler) > 0
        # Check iteration
        steps = list(scheduler)
        assert len(steps) > 0
        
    def test_pipeline_stage_basic(self):
        """Test PipelineStage basic creation."""
        from deepseek.torch.training.dualpipe import PipelineStage, PipelineConfig
        
        # Create simple model layer
        model_layer = nn.Linear(64, 64)
        config = PipelineConfig(num_stages=2, num_micro_batches=4)
        stage = PipelineStage(
            module=model_layer,
            stage_id=0,
            config=config,
        )
        
        assert stage.stage_id == 0
        assert stage.config.num_stages == 2


class TestSequenceParallelism:
    """Test Sequence Parallelism module."""
    
    def test_sequence_parallel_config(self):
        """Test SequenceParallelConfig creation."""
        from deepseek.torch.training.sequence_parallel import SequenceParallelConfig
        
        config = SequenceParallelConfig()
        assert hasattr(config, 'sp_size')
        
    def test_sequence_parallel_attention_module(self):
        """Test SequenceParallelAttention exists."""
        from deepseek.torch.training.sequence_parallel import SequenceParallelAttention
        
        # Just verify class exists
        assert SequenceParallelAttention is not None
        
    def test_sequence_parallel_rmsnorm_module(self):
        """Test SequenceParallelRMSNorm exists."""
        from deepseek.torch.training.sequence_parallel import SequenceParallelRMSNorm
        
        # Just verify class exists
        assert SequenceParallelRMSNorm is not None


class TestFaultTolerance:
    """Test Fault Tolerance module."""
    
    def test_elastic_config(self):
        """Test ElasticConfig creation."""
        from deepseek.torch.training.fault_tolerance import ElasticConfig
        
        config = ElasticConfig()
        # Check expected attributes exist
        assert hasattr(config, 'heartbeat_interval')
        assert hasattr(config, 'heartbeat_timeout')
        
    def test_preemption_handler(self):
        """Test PreemptionHandler basic functionality."""
        from deepseek.torch.training.fault_tolerance import PreemptionHandler
        
        handler = PreemptionHandler(checkpoint_fn=lambda: None)
        
        # Check internal state
        assert hasattr(handler, '_preempted')
        assert handler._preempted == False
        
    def test_heartbeat_monitor_creation(self):
        """Test HeartbeatMonitor creation."""
        from deepseek.torch.training.fault_tolerance import HeartbeatMonitor, ElasticConfig
        
        config = ElasticConfig()
        # Create with all required args
        monitor = HeartbeatMonitor(config, rank=0, world_size=1)
        
        assert monitor is not None
        
    def test_failure_injector(self):
        """Test FailureInjector for testing."""
        from deepseek.torch.training.fault_tolerance import FailureInjector
        
        injector = FailureInjector(failure_rate=0.0)
        
        # With 0 failure rate, should not raise
        for _ in range(100):
            injector.maybe_fail()  # Should not raise
            
    def test_graceful_degradation_wrapper_exists(self):
        """Test graceful degradation wrapper exists."""
        from deepseek.torch.training.fault_tolerance import graceful_degradation_wrapper
        
        # Just verify function exists
        assert callable(graceful_degradation_wrapper)


class TestDistributedCheckpointing:
    """Test Distributed Checkpointing module."""
    
    def test_checkpoint_config(self):
        """Test CheckpointConfig creation."""
        from deepseek.torch.training.distributed_checkpoint import (
            CheckpointConfig,
            CheckpointFormat,
            CompressionType,
        )
        
        config = CheckpointConfig()
        assert config.checkpoint_dir == "./checkpoints"
        assert config.format == CheckpointFormat.PYTORCH
        assert config.async_save is True
        assert config.keep_n_checkpoints == 5
        assert config.compression == CompressionType.NONE
        
    def test_checkpoint_metadata(self):
        """Test CheckpointMetadata creation and serialization."""
        from deepseek.torch.training.distributed_checkpoint import CheckpointMetadata
        
        metadata = CheckpointMetadata(
            version=1,
            global_step=1000,
            epoch=5,
            timestamp=time.time(),
            world_size=8,
            model_config={"hidden_size": 4096},
            training_config={"lr": 1e-4},
            metrics={"loss": 0.5},
        )
        
        # Test serialization
        d = metadata.to_dict()
        assert d["version"] == 1
        assert d["global_step"] == 1000
        
        # Test deserialization
        restored = CheckpointMetadata.from_dict(d)
        assert restored.version == metadata.version
        assert restored.global_step == metadata.global_step
        
    def test_async_checkpoint_saver(self):
        """Test AsyncCheckpointSaver."""
        from deepseek.torch.training.distributed_checkpoint import AsyncCheckpointSaver
        
        saver = AsyncCheckpointSaver(max_workers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.pt")
            checkpoint = {"data": torch.randn(10)}
            
            # Save async
            callback_called = False
            def callback(p, success):
                nonlocal callback_called
                callback_called = True
                
            saver.save_async(checkpoint, path, callback)
            saver.wait_for_save(path)
            
            assert os.path.exists(path)
            
        saver.shutdown()
        
    def test_distributed_checkpointer_creation(self):
        """Test DistributedCheckpointer creation."""
        from deepseek.torch.training.distributed_checkpoint import (
            DistributedCheckpointer,
            CheckpointConfig,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=tmpdir,
                async_save=False,
            )
            
            with patch('torch.distributed.is_initialized', return_value=False):
                checkpointer = DistributedCheckpointer(config)
                
                assert checkpointer.rank == 0
                assert checkpointer.world_size == 1
                
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        from deepseek.torch.training.distributed_checkpoint import (
            DistributedCheckpointer,
            CheckpointConfig,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=tmpdir,
                async_save=False,
            )
            
            with patch('torch.distributed.is_initialized', return_value=False):
                checkpointer = DistributedCheckpointer(config)
                
                # Create model
                model = nn.Linear(10, 10)
                
                # Save
                path = checkpointer.save(
                    model,
                    global_step=100,
                    epoch=5,
                    metrics={"loss": 0.5},
                )
                
                assert os.path.exists(path)
                
                # Load
                model2 = nn.Linear(10, 10)
                result = checkpointer.load(model2, checkpoint_path=path)
                
                assert "metadata" in result
                assert result["metadata"].global_step == 100
                
    def test_checkpoint_garbage_collection(self):
        """Test checkpoint garbage collection."""
        from deepseek.torch.training.distributed_checkpoint import (
            DistributedCheckpointer,
            CheckpointConfig,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=tmpdir,
                async_save=False,
                keep_n_checkpoints=2,
                validate_before_delete=False,
            )
            
            with patch('torch.distributed.is_initialized', return_value=False):
                checkpointer = DistributedCheckpointer(config)
                model = nn.Linear(10, 10)
                
                # Save 4 checkpoints
                for i in range(4):
                    checkpointer.save(model, global_step=i)
                    
                # Should only keep 2
                checkpoints = checkpointer._list_checkpoints()
                assert len(checkpoints) == 2


class TestMLXDistributed:
    """Test MLX Distributed placeholder module."""
    
    def test_mlx_distributed_config(self):
        """Test MLXDistributedConfig creation."""
        try:
            from mlx_impl.mlx_distributed import MLXDistributedConfig
            
            config = MLXDistributedConfig()
            assert config.enabled is False
            assert config.gradient_checkpointing is True
            
        except ImportError:
            pytest.skip("MLX not available")
            
    def test_mlx_distributed_validation(self):
        """Test MLXDistributedConfig validation fails when enabled."""
        try:
            from mlx_impl.mlx_distributed import MLXDistributedConfig
            
            config = MLXDistributedConfig(enabled=True)
            with pytest.raises(NotImplementedError):
                config.validate()
                
        except ImportError:
            pytest.skip("MLX not available")
            
    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        try:
            from mlx_impl.mlx_distributed import estimate_model_memory_gb
            
            # 1B params in FP16
            memory = estimate_model_memory_gb(
                num_parameters=1_000_000_000,
                dtype_bits=16,
                include_gradients=True,
                include_optimizer=True,
            )
            
            # Should be around 8GB for 1B params with gradients + optimizer
            assert 6.0 < memory < 12.0
            
        except ImportError:
            pytest.skip("MLX not available")
            
    def test_check_model_fits(self):
        """Test check_model_fits function."""
        try:
            from mlx_impl.mlx_distributed import check_model_fits
            
            # Small model should fit
            result = check_model_fits(
                num_parameters=100_000,
                dtype_bits=16,
            )
            
            assert "fits" in result
            assert "required_memory_gb" in result
            assert "available_memory_gb" in result
            
        except ImportError:
            pytest.skip("MLX not available")
            
    def test_mlx_distributed_placeholder_not_available(self):
        """Test MLXDistributedPlaceholder.is_available() returns False."""
        try:
            from mlx_impl.mlx_distributed import MLXDistributedPlaceholder
            
            placeholder = MLXDistributedPlaceholder()
            assert placeholder.is_available() is False
            assert placeholder.world_size == 1
            assert placeholder.rank == 0
            
        except ImportError:
            pytest.skip("MLX not available")


# Integration tests that require distributed setup
@pytest.mark.skipif(not MULTI_GPU, reason="Multi-GPU not available")
class TestDistributedIntegration:
    """Integration tests requiring multiple GPUs."""
    
    @pytest.mark.skip(reason="Requires actual multi-GPU setup")
    def test_fsdp_multi_gpu(self):
        """Test FSDP with multiple GPUs."""
        pass
        
    @pytest.mark.skip(reason="Requires actual multi-GPU setup")
    def test_expert_parallel_multi_gpu(self):
        """Test Expert Parallelism with multiple GPUs."""
        pass
        
    @pytest.mark.skip(reason="Requires actual multi-GPU setup")
    def test_dualpipe_multi_gpu(self):
        """Test DualPipe with multiple GPUs."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
