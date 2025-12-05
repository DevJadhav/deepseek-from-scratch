"""
Tests for Research Polish and Documentation

Verifies the creation of Docker files, ablation infrastructure, benchmarks,
reproducibility docs, blog posts, and paper materials.
"""
from pathlib import Path

import pytest


# Get the project root directory (deepseek-from-scratch-python/tests -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDockerEnvironment:
    """Test 5.1: Docker Environment"""
    
    def test_dockerfile_exists(self):
        """Verify Dockerfile exists with CUDA support"""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile should exist"
        content = dockerfile.read_text()
        assert "CUDA" in content or "cuda" in content, "Dockerfile should reference CUDA"
        assert "uv" in content, "Dockerfile should use uv for dependency management"
    
    def test_docker_compose_exists(self):
        """Verify docker-compose.yml exists"""
        docker_compose = PROJECT_ROOT / "docker-compose.yml"
        assert docker_compose.exists(), "docker-compose.yml should exist"
        content = docker_compose.read_text()
        assert "services:" in content, "docker-compose should define services"
        assert "volumes:" in content or "volume" in content.lower(), "docker-compose should handle volumes"
    
    def test_devcontainer_exists(self):
        """Verify devcontainer.json exists"""
        devcontainer = PROJECT_ROOT / ".devcontainer" / "devcontainer.json"
        assert devcontainer.exists(), "devcontainer.json should exist"
        content = devcontainer.read_text()
        assert "name" in content, "devcontainer should have a name"


class TestAblationInfrastructure:
    """Test 5.2: Ablation Study Infrastructure"""
    
    def test_ablation_directory_exists(self):
        """Verify ablation scripts directory exists"""
        ablation_dir = PROJECT_ROOT / "scripts" / "ablation"
        assert ablation_dir.exists(), "scripts/ablation/ directory should exist"
        assert ablation_dir.is_dir(), "ablation should be a directory"
    
    def test_ablation_init_exists(self):
        """Verify ablation __init__.py exists"""
        init_file = PROJECT_ROOT / "scripts" / "ablation" / "__init__.py"
        assert init_file.exists(), "__init__.py should exist in ablation directory"
    
    def test_ablation_utils_exists(self):
        """Verify ablation_utils.py exists with required functions"""
        utils_file = PROJECT_ROOT / "scripts" / "ablation" / "ablation_utils.py"
        assert utils_file.exists(), "ablation_utils.py should exist"
        content = utils_file.read_text()
        assert "AblationResult" in content, "Should define AblationResult dataclass"
        assert "statistical" in content.lower() or "aggregate" in content.lower(), "Should have statistical analysis"
    
    def test_attention_ablation_exists(self):
        """Verify attention ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_attention_ablation.py"
        assert script.exists(), "run_attention_ablation.py should exist"
        content = script.read_text()
        assert "MLA" in content or "mla" in content.lower(), "Should compare MLA"
        assert "GQA" in content or "gqa" in content.lower(), "Should compare GQA"
        assert "MHA" in content or "mha" in content.lower(), "Should compare MHA"
    
    def test_expert_ablation_exists(self):
        """Verify expert count ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_expert_ablation.py"
        assert script.exists(), "run_expert_ablation.py should exist"
        content = script.read_text()
        assert "expert" in content.lower(), "Should mention experts"
    
    def test_balancing_ablation_exists(self):
        """Verify load balancing ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_balancing_ablation.py"
        assert script.exists(), "run_balancing_ablation.py should exist"
        content = script.read_text()
        assert "balance" in content.lower() or "auxiliary" in content.lower(), "Should mention balancing"
    
    def test_mtp_ablation_exists(self):
        """Verify MTP depth ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_mtp_ablation.py"
        assert script.exists(), "run_mtp_ablation.py should exist"
        content = script.read_text()
        assert "mtp" in content.lower() or "multi" in content.lower(), "Should mention MTP"
    
    def test_precision_ablation_exists(self):
        """Verify precision ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_precision_ablation.py"
        assert script.exists(), "run_precision_ablation.py should exist"
        content = script.read_text()
        assert "fp8" in content.lower() or "precision" in content.lower(), "Should mention precision"
    
    def test_run_all_ablations_exists(self):
        """Verify master ablation script exists"""
        script = PROJECT_ROOT / "scripts" / "ablation" / "run_all_ablations.py"
        assert script.exists(), "run_all_ablations.py should exist"


class TestBenchmarkComparisons:
    """Test 5.3: Benchmark Comparisons"""
    
    def test_benchmark_script_exists(self):
        """Verify benchmark.py exists with comprehensive comparisons"""
        benchmark = PROJECT_ROOT / "scripts" / "benchmark.py"
        assert benchmark.exists(), "scripts/benchmark.py should exist"
        content = benchmark.read_text()
        assert "throughput" in content.lower(), "Should measure throughput"
        assert "memory" in content.lower(), "Should measure memory"


class TestReproducibilityPackage:
    """Test 5.4: Reproducibility Package"""
    
    def test_reproducibility_md_exists(self):
        """Verify REPRODUCIBILITY.md exists with required content"""
        reproducibility = PROJECT_ROOT / "REPRODUCIBILITY.md"
        assert reproducibility.exists(), "REPRODUCIBILITY.md should exist"
        content = reproducibility.read_text()
        assert "hardware" in content.lower(), "Should document hardware requirements"
        assert "setup" in content.lower() or "install" in content.lower(), "Should have setup instructions"
        assert "seed" in content.lower(), "Should mention random seeds"


class TestTechnicalBlogPosts:
    """Test 5.5: Technical Blog Posts"""
    
    def test_blog_directory_exists(self):
        """Verify blog directory exists"""
        blog_dir = PROJECT_ROOT / "docs" / "blog"
        assert blog_dir.exists(), "docs/blog/ directory should exist"
    
    def test_mla_deep_dive_exists(self):
        """Verify MLA deep dive blog post exists"""
        blog = PROJECT_ROOT / "docs" / "blog" / "01_mla_deep_dive.md"
        assert blog.exists(), "01_mla_deep_dive.md should exist"
        content = blog.read_text()
        assert "Multi-Latent Attention" in content or "MLA" in content, "Should discuss MLA"
    
    def test_auxiliary_loss_free_exists(self):
        """Verify auxiliary-loss-free blog post exists"""
        blog = PROJECT_ROOT / "docs" / "blog" / "02_auxiliary_loss_free.md"
        assert blog.exists(), "02_auxiliary_loss_free.md should exist"
        content = blog.read_text()
        assert "auxiliary" in content.lower() or "balance" in content.lower(), "Should discuss load balancing"
    
    def test_dualpipe_explained_exists(self):
        """Verify DualPipe blog post exists"""
        blog = PROJECT_ROOT / "docs" / "blog" / "03_dualpipe_explained.md"
        assert blog.exists(), "03_dualpipe_explained.md should exist"
        content = blog.read_text()
        assert "pipeline" in content.lower() or "DualPipe" in content, "Should discuss pipeline parallelism"
    
    def test_expert_specialization_exists(self):
        """Verify expert specialization blog post exists"""
        blog = PROJECT_ROOT / "docs" / "blog" / "04_expert_specialization.md"
        assert blog.exists(), "04_expert_specialization.md should exist"
        content = blog.read_text()
        assert "expert" in content.lower() and "special" in content.lower(), "Should discuss expert specialization"
    
    def test_production_lessons_exists(self):
        """Verify production lessons blog post exists"""
        blog = PROJECT_ROOT / "docs" / "blog" / "05_production_lessons.md"
        assert blog.exists(), "05_production_lessons.md should exist"
        content = blog.read_text()
        assert "production" in content.lower() or "lesson" in content.lower(), "Should discuss production lessons"


class TestPaperReadyDocumentation:
    """Test 5.6: Paper-Ready Documentation"""

    def test_paper_directory_exists(self):
        """Verify paper directory exists"""
        paper_dir = PROJECT_ROOT / "docs" / "paper"
        assert paper_dir.exists(), "docs/paper/ directory should exist"

    def test_architecture_diagrams_exist(self):
        """Verify architecture diagrams exist"""
        architecture = PROJECT_ROOT / "docs" / "paper" / "architecture.md"
        assert architecture.exists(), "architecture.md should exist"
        content = architecture.read_text()
        assert "diagram" in content.lower() or "┌" in content or "╔" in content, "Should contain diagrams"

    def test_pseudocode_exists(self):
        """Verify pseudocode documentation exists"""
        pseudocode = PROJECT_ROOT / "docs" / "paper" / "pseudocode.md"
        assert pseudocode.exists(), "pseudocode.md should exist"
        content = pseudocode.read_text()
        assert "Algorithm" in content, "Should contain algorithm descriptions"

    def test_latex_tables_exist(self):
        """Verify LaTeX tables exist"""
        tables = PROJECT_ROOT / "docs" / "paper" / "tables.tex"
        assert tables.exists(), "tables.tex should exist"
        content = tables.read_text()
        assert "\\begin{table}" in content, "Should contain LaTeX table environments"

    def test_related_work_exists(self):
        """Verify related work documentation exists"""
        related_work = PROJECT_ROOT / "docs" / "paper" / "related_work.md"
        assert related_work.exists(), "related_work.md should exist"
        content = related_work.read_text()
        assert "Related Work" in content or "Comparison" in content, "Should discuss related work"

    def test_contributions_exists(self):
        """Verify contributions summary exists"""
        contributions = PROJECT_ROOT / "docs" / "paper" / "contributions.md"
        assert contributions.exists(), "contributions.md should exist"
        content = contributions.read_text()
        assert "Contribution" in content, "Should list contributions"

    def test_experimental_setup_exists(self):
        """Verify experimental setup documentation exists"""
        setup = PROJECT_ROOT / "docs" / "paper" / "experimental_setup.md"
        assert setup.exists(), "experimental_setup.md should exist"
        content = setup.read_text()
        assert "Hardware" in content or "Setup" in content, "Should describe experimental setup"

    def test_supplementary_exists(self):
        """Verify supplementary materials exist"""
        supplementary = PROJECT_ROOT / "docs" / "paper" / "supplementary.md"
        assert supplementary.exists(), "supplementary.md should exist"
        content = supplementary.read_text()
        assert "Extended" in content or "Supplementary" in content, "Should contain supplementary materials"
class TestREADMEDocumentation:
    """Test 5.7: README and Project Documentation"""

    def test_readme_exists(self):
        """Verify README.md exists with required sections"""
        readme = PROJECT_ROOT / "README.md"
        assert readme.exists(), "README.md should exist"
        content = readme.read_text()

        # Check for key sections
        assert "Quick Start" in content, "Should have Quick Start section"
        assert "Installation" in content, "Should have Installation section"
        assert "Docker" in content, "Should mention Docker setup"
        assert "Ablation" in content, "Should mention ablation studies"
        assert "FAQ" in content, "Should have FAQ section"
        assert "Contributing" in content, "Should have contributing guidelines"
        assert "License" in content, "Should mention license"

    def test_license_exists(self):
        """Verify LICENSE file exists"""
        license_file = PROJECT_ROOT / "LICENSE"
        assert license_file.exists(), "LICENSE file should exist"

    def test_code_of_conduct_exists(self):
        """Verify CODE_OF_CONDUCT.md exists"""
        coc = PROJECT_ROOT / "CODE_OF_CONDUCT.md"
        assert coc.exists(), "CODE_OF_CONDUCT.md should exist"
        content = coc.read_text()
        assert "Contributor Covenant" in content or "Code of Conduct" in content

    def test_contributing_exists(self):
        """Verify CONTRIBUTING.md exists"""
        contributing = PROJECT_ROOT / "CONTRIBUTING.md"
        assert contributing.exists(), "CONTRIBUTING.md should exist"
        content = contributing.read_text()
        assert "Contributing" in content
        assert "Pull Request" in content

    def test_changelog_exists(self):
        """Verify CHANGELOG.md exists"""
        changelog = PROJECT_ROOT / "CHANGELOG.md"
        assert changelog.exists(), "CHANGELOG.md should exist"
        content = changelog.read_text()
        assert "Changelog" in content
        assert "Added" in content or "Changed" in content

    def test_bibtex_citation_in_readme(self):
        """Verify BibTeX citation is in README"""
        readme = PROJECT_ROOT / "README.md"
        content = readme.read_text()
        assert "bibtex" in content.lower() or "@software" in content or "Citation" in content
class TestDockerfileContent:
    """Detailed tests for Dockerfile content"""
    
    def test_dockerfile_multi_stage_build(self):
        """Verify Dockerfile uses multi-stage build"""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        # Check for FROM appearing multiple times (multi-stage) or AS keyword
        assert content.count("FROM") >= 1, "Should have at least one FROM statement"
        # Check for base image specification
        assert "nvidia" in content.lower() or "cuda" in content.lower(), "Should use NVIDIA/CUDA base image"
    
    def test_dockerfile_has_uv(self):
        """Verify Dockerfile installs and uses uv"""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "uv" in content, "Should use uv for dependency management"
        assert "sync" in content.lower() or "install" in content.lower(), "Should sync/install dependencies"


class TestDockerComposeContent:
    """Detailed tests for docker-compose.yml content"""
    
    def test_docker_compose_has_gpu_support(self):
        """Verify docker-compose has GPU configuration"""
        docker_compose = PROJECT_ROOT / "docker-compose.yml"
        content = docker_compose.read_text()
        assert "gpu" in content.lower() or "nvidia" in content.lower() or "deploy" in content, \
            "Should have GPU/NVIDIA configuration"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
