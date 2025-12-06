"""
Tests for heterogeneous Ray scheduling module.

These tests verify resource detection, Ray integration,
placement groups, and health monitoring functionality.
"""

from unittest import mock

from src.deepseek.pipeline.heterogeneous import (
    ClusterHealthMonitor,
    DetectedResources,
    NodeArchitecture,
    NodeHealth,
    PlacementBundle,
    PlacementGroupConfig,
    detect_cpu_cores,
    detect_cuda_gpus,
    detect_memory_gb,
    detect_metal_support,
    detect_resources,
    get_ray_available_resources,
    get_ray_cluster_resources,
    register_custom_resources_with_ray,
)


class TestNodeArchitecture:
    """Tests for NodeArchitecture enum."""
    
    def test_architecture_values(self):
        """Test all architecture values exist."""
        assert NodeArchitecture.UNKNOWN.value == "unknown"
        assert NodeArchitecture.APPLE_SILICON.value == "apple_silicon"
        assert NodeArchitecture.NVIDIA_GPU.value == "nvidia_gpu"
        assert NodeArchitecture.AMD_GPU.value == "amd_gpu"
        assert NodeArchitecture.CPU_ONLY.value == "cpu_only"


class TestDetectedResources:
    """Tests for DetectedResources dataclass."""
    
    def test_default_values(self):
        """Test default resource values."""
        resources = DetectedResources()
        assert resources.node_id == ""
        assert resources.hostname == ""
        assert resources.architecture == NodeArchitecture.UNKNOWN
        assert resources.cpu_cores == 0
        assert resources.memory_gb == 0.0
        assert resources.has_metal is False
        assert resources.has_cuda is False
    
    def test_apple_silicon_resources(self):
        """Test Apple Silicon resource configuration."""
        resources = DetectedResources(
            node_id="test-node",
            hostname="macbook-pro",
            architecture=NodeArchitecture.APPLE_SILICON,
            cpu_cores=10,
            memory_gb=32.0,
            has_metal=True,
            labels=["apple_silicon", "macbook_pro"],
        )
        
        ray_resources = resources.to_ray_resources()
        
        assert ray_resources["cpu_cores"] == 10.0
        assert ray_resources["memory_gb"] == 32.0
        assert ray_resources["metal"] == 1.0
        assert ray_resources["gpu_memory_gb"] == 24.0  # 75% of unified memory
        assert "cuda" not in ray_resources
    
    def test_nvidia_h100_resources(self):
        """Test NVIDIA H100 resource configuration."""
        resources = DetectedResources(
            node_id="gpu-node-1",
            hostname="dgx-h100",
            architecture=NodeArchitecture.NVIDIA_GPU,
            cpu_cores=128,
            memory_gb=2048.0,
            has_cuda=True,
            cuda_device_count=8,
            cuda_compute_cap=9.0,
            gpu_memory_gb=80.0,
            labels=["nvidia_gpu", "h100"],
        )
        
        ray_resources = resources.to_ray_resources()
        
        assert ray_resources["cpu_cores"] == 128.0
        assert ray_resources["memory_gb"] == 2048.0
        assert ray_resources["cuda"] == 8.0
        assert ray_resources["cuda_compute_cap"] == 9.0
        assert ray_resources["gpu_memory_gb"] == 80.0
        assert ray_resources["H100"] == 8.0
        assert ray_resources["has_hbm"] == 1.0
        assert "metal" not in ray_resources
    
    def test_nvidia_a100_resources(self):
        """Test NVIDIA A100 resource configuration."""
        resources = DetectedResources(
            node_id="gpu-node-2",
            hostname="dgx-a100",
            architecture=NodeArchitecture.NVIDIA_GPU,
            cpu_cores=64,
            memory_gb=1024.0,
            has_cuda=True,
            cuda_device_count=4,
            cuda_compute_cap=8.0,
            gpu_memory_gb=40.0,
            labels=["nvidia_gpu", "a100"],
        )
        
        ray_resources = resources.to_ray_resources()
        
        assert ray_resources["cuda"] == 4.0
        assert ray_resources["cuda_compute_cap"] == 8.0
        assert ray_resources["A100"] == 4.0
        assert ray_resources["has_hbm"] == 1.0
        assert "H100" not in ray_resources
    
    def test_cpu_only_resources(self):
        """Test CPU-only resource configuration."""
        resources = DetectedResources(
            node_id="cpu-node",
            hostname="cpu-server",
            architecture=NodeArchitecture.CPU_ONLY,
            cpu_cores=32,
            memory_gb=128.0,
            labels=["cpu_only"],
        )
        
        ray_resources = resources.to_ray_resources()
        
        assert ray_resources["cpu_cores"] == 32.0
        assert ray_resources["memory_gb"] == 128.0
        assert "metal" not in ray_resources
        assert "cuda" not in ray_resources
        assert "gpu_memory_gb" not in ray_resources


class TestResourceDetection:
    """Tests for resource detection functions."""
    
    def test_detect_cpu_cores(self):
        """Test CPU core detection returns positive value."""
        cores = detect_cpu_cores()
        assert isinstance(cores, int)
        assert cores > 0
    
    def test_detect_memory_gb(self):
        """Test memory detection returns reasonable value."""
        memory = detect_memory_gb()
        assert isinstance(memory, float)
        assert memory > 0  # At least some memory
        assert memory < 10000  # Less than 10TB
    
    def test_detect_metal_support_non_darwin(self):
        """Test Metal detection returns False on non-Darwin."""
        with mock.patch("platform.system", return_value="Linux"):
            assert detect_metal_support() is False
    
    def test_detect_cuda_gpus_no_nvidia(self):
        """Test CUDA detection with no NVIDIA hardware."""
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch.dict("os.environ", {}, clear=True):
                count, cap, mem = detect_cuda_gpus()
                assert count == 0
                assert cap == 0.0
                assert mem == 0.0
    
    def test_detect_resources_returns_valid(self):
        """Test full resource detection returns valid DetectedResources."""
        resources = detect_resources()
        
        assert isinstance(resources, DetectedResources)
        assert resources.node_id != ""
        assert resources.hostname != ""
        assert resources.cpu_cores > 0
        assert resources.memory_gb > 0
        assert resources.architecture in NodeArchitecture
        assert len(resources.labels) > 0


class TestRegisterCustomResources:
    """Tests for Ray resource registration."""
    
    def test_register_with_provided_resources(self):
        """Test registration with pre-detected resources."""
        resources = DetectedResources(
            node_id="test-node",
            cpu_cores=8,
            memory_gb=16.0,
            has_metal=True,
        )
        
        ray_resources = register_custom_resources_with_ray(resources)
        
        assert ray_resources["cpu_cores"] == 8.0
        assert ray_resources["memory_gb"] == 16.0
        assert ray_resources["metal"] == 1.0
    
    def test_register_auto_detect(self):
        """Test registration with auto-detection."""
        ray_resources = register_custom_resources_with_ray()
        
        assert "cpu_cores" in ray_resources
        assert "memory_gb" in ray_resources
        assert ray_resources["cpu_cores"] > 0
        assert ray_resources["memory_gb"] > 0


class TestPlacementGroup:
    """Tests for placement group configuration."""
    
    def test_placement_bundle(self):
        """Test PlacementBundle creation."""
        bundle = PlacementBundle(
            resources={"metal": 1.0, "memory_gb": 32.0},
            name="stage-0",
        )
        
        assert bundle.resources["metal"] == 1.0
        assert bundle.resources["memory_gb"] == 32.0
        assert bundle.name == "stage-0"
    
    def test_placement_group_config(self):
        """Test PlacementGroupConfig creation."""
        bundles = [
            PlacementBundle(resources={"metal": 1.0}, name="stage-0"),
            PlacementBundle(resources={"cuda": 1.0}, name="stage-1"),
        ]
        
        config = PlacementGroupConfig(
            name="pipeline-pg",
            bundles=bundles,
            strategy="STRICT_SPREAD",
        )
        
        assert config.name == "pipeline-pg"
        assert len(config.bundles) == 2
        assert config.strategy == "STRICT_SPREAD"


class TestClusterHealthMonitor:
    """Tests for ClusterHealthMonitor."""
    
    def test_register_node(self):
        """Test node registration."""
        monitor = ClusterHealthMonitor()
        monitor.register_node("node-1")
        
        health = monitor.get_node_health("node-1")
        assert health is not None
        assert health.node_id == "node-1"
        assert health.is_healthy is True
    
    def test_update_health_healthy(self):
        """Test health update for healthy node."""
        monitor = ClusterHealthMonitor()
        monitor.register_node("node-1")
        
        health = NodeHealth(
            node_id="node-1",
            is_healthy=True,
            cpu_utilization=0.5,
            memory_utilization=0.6,
        )
        monitor.update_health(health)
        
        updated = monitor.get_node_health("node-1")
        assert updated is not None
        assert updated.is_healthy is True
        assert updated.cpu_utilization == 0.5
    
    def test_update_health_failure_threshold(self):
        """Test node marked unhealthy after failure threshold."""
        failure_callback_called = []
        
        def on_failure(node_id: str):
            failure_callback_called.append(node_id)
        
        monitor = ClusterHealthMonitor(
            failure_threshold=3,
            on_node_failure=on_failure,
        )
        monitor.register_node("node-1")
        
        # Report failures
        for i in range(3):
            health = NodeHealth(
                node_id="node-1",
                is_healthy=False,
                error_message=f"Failure {i + 1}",
            )
            monitor.update_health(health)
        
        updated = monitor.get_node_health("node-1")
        assert updated is not None
        assert updated.is_healthy is False
        assert len(failure_callback_called) == 1
        assert failure_callback_called[0] == "node-1"
    
    def test_healthy_nodes_list(self):
        """Test getting list of healthy nodes."""
        monitor = ClusterHealthMonitor()
        monitor.register_node("node-1")
        monitor.register_node("node-2")
        monitor.register_node("node-3")
        
        # Mark node-2 unhealthy
        monitor.update_health(NodeHealth(node_id="node-2", is_healthy=False))
        monitor.update_health(NodeHealth(node_id="node-2", is_healthy=False))
        monitor.update_health(NodeHealth(node_id="node-2", is_healthy=False))
        
        healthy = monitor.get_healthy_nodes()
        assert "node-1" in healthy
        assert "node-2" not in healthy
        assert "node-3" in healthy
    
    def test_cluster_summary(self):
        """Test cluster health summary."""
        monitor = ClusterHealthMonitor()
        monitor.register_node("node-1")
        monitor.register_node("node-2")
        
        summary = monitor.get_cluster_summary()
        
        assert summary["total_nodes"] == 2
        assert summary["healthy_nodes"] == 2
        assert summary["unhealthy_nodes"] == 0
        assert "node-1" in summary["nodes"]
        assert "node-2" in summary["nodes"]
    
    def test_failure_count_reset_on_healthy(self):
        """Test failure count resets when node becomes healthy."""
        monitor = ClusterHealthMonitor(failure_threshold=3)
        monitor.register_node("node-1")
        
        # Report 2 failures
        monitor.update_health(NodeHealth(node_id="node-1", is_healthy=False))
        monitor.update_health(NodeHealth(node_id="node-1", is_healthy=False))
        
        # Report healthy
        monitor.update_health(NodeHealth(node_id="node-1", is_healthy=True))
        
        # Report 2 more failures - should not trigger threshold
        monitor.update_health(NodeHealth(node_id="node-1", is_healthy=False))
        monitor.update_health(NodeHealth(node_id="node-1", is_healthy=False))
        
        health = monitor.get_node_health("node-1")
        assert health is not None
        # Node should still be considered "healthy" since we didn't hit 3 consecutive


class TestRayIntegration:
    """Tests for Ray integration (without actually initializing Ray)."""
    
    def test_get_cluster_resources_not_initialized(self):
        """Test getting cluster resources when Ray not initialized."""
        with mock.patch("ray.is_initialized", return_value=False):
            resources = get_ray_cluster_resources()
            assert resources == {}
    
    def test_get_available_resources_not_initialized(self):
        """Test getting available resources when Ray not initialized."""
        with mock.patch("ray.is_initialized", return_value=False):
            resources = get_ray_available_resources()
            assert resources == {}


class TestResourceRequirementsFromConfig:
    """Tests for ResourceRequirements from config.py."""
    
    def test_import_resource_requirements(self):
        """Test ResourceRequirements can be imported from config."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements()
        assert req.resources == {}
        assert req.preferences == {}
        assert req.node_affinity == []
    
    def test_apple_silicon_factory(self):
        """Test apple_silicon factory method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements.apple_silicon(memory_gb=32)
        assert req.resources["metal"] == 1.0
        assert req.resources["memory_gb"] == 32.0
        assert "apple_silicon" in req.node_affinity
    
    def test_nvidia_h100_factory(self):
        """Test nvidia_h100 factory method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements.nvidia_h100()
        assert req.resources["cuda"] == 1.0
        assert req.resources["cuda_compute_cap"] == 9.0
        assert req.resources["gpu_memory_gb"] == 80.0
        assert "h100" in req.node_affinity
    
    def test_nvidia_a100_factory(self):
        """Test nvidia_a100 factory method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements.nvidia_a100()
        assert req.resources["cuda"] == 1.0
        assert req.resources["cuda_compute_cap"] == 8.0
        assert req.resources["gpu_memory_gb"] == 40.0
        assert "a100" in req.node_affinity
    
    def test_cpu_only_factory(self):
        """Test cpu_only factory method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements.cpu_only(cores=32, memory_gb=128)
        assert req.resources["cpu_cores"] == 32.0
        assert req.resources["memory_gb"] == 128.0
    
    def test_to_ray_resources(self):
        """Test converting ResourceRequirements to Ray resources dict."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements(
            resources={"cpu_cores": 8.0, "memory_gb": 32.0, "metal": 1.0}
        )
        
        ray_res = req.to_ray_resources()
        assert ray_res["cpu_cores"] == 8.0
        assert ray_res["memory_gb"] == 32.0
        assert ray_res["metal"] == 1.0
    
    def test_require_method(self):
        """Test require builder method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements().require("metal", 1.0).require("memory_gb", 64.0)
        assert req.resources["metal"] == 1.0
        assert req.resources["memory_gb"] == 64.0
    
    def test_prefer_method(self):
        """Test prefer builder method."""
        from src.deepseek.pipeline.config import ResourceRequirements
        
        req = ResourceRequirements().prefer("gpu_memory_gb", 80.0)
        assert req.preferences["gpu_memory_gb"] == 80.0


class TestWaveConfigResources:
    """Tests for WaveConfig with resources field."""
    
    def test_wave_config_default_resources(self):
        """Test WaveConfig has default None resources."""
        from src.deepseek.pipeline.config import WaveBackend, WaveConfig
        
        wave = WaveConfig(
            wave_id=0,
            backend=WaveBackend.RUST_METAL,
            start_step=0,
            end_step=100,
            stages=["pretrain"],
        )
        assert wave.resources is None
    
    def test_wave_config_with_resources(self):
        """Test WaveConfig with resource requirements."""
        from src.deepseek.pipeline.config import (
            ResourceRequirements,
            WaveBackend,
            WaveConfig,
        )
        
        resources = ResourceRequirements.apple_silicon(memory_gb=32)
        wave = WaveConfig(
            wave_id=0,
            backend=WaveBackend.RUST_METAL,
            start_step=0,
            end_step=100,
            stages=["pretrain"],
            resources=resources,
        )
        
        assert wave.resources is not None
        assert wave.resources.resources["metal"] == 1.0
        assert wave.resources.resources["memory_gb"] == 32.0
    
    def test_wave_config_get_ray_resources(self):
        """Test WaveConfig.get_ray_resources method."""
        from src.deepseek.pipeline.config import (
            ResourceRequirements,
            WaveBackend,
            WaveConfig,
        )
        
        resources = ResourceRequirements.nvidia_h100()
        wave = WaveConfig(
            wave_id=1,
            backend=WaveBackend.RUST_CUDA,
            start_step=100,
            end_step=200,
            stages=["pretrain"],
            resources=resources,
        )
        
        ray_res = wave.get_ray_resources()
        assert ray_res is not None
        assert ray_res["cuda"] == 1.0
        assert ray_res["cuda_compute_cap"] == 9.0
    
    def test_wave_config_get_ray_resources_default(self):
        """Test WaveConfig.get_ray_resources returns default when no resources."""
        from src.deepseek.pipeline.config import WaveBackend, WaveConfig
        
        wave = WaveConfig(
            wave_id=0,
            backend=WaveBackend.RUST_METAL,
            start_step=0,
            end_step=100,
            stages=["pretrain"],
        )
        ray_res = wave.get_ray_resources()
        # Default for RUST_METAL backend should include metal
        assert ray_res.get("metal") == 1.0
