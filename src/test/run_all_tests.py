#!/usr/bin/env python3
"""
Run all tests for the AudioSet data pipeline.

This script runs the complete test suite including:
- Data processor tests
- Dataset and sampler tests  
- Integration tests
- Performance benchmarks
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_test_module(module_name: str, description: str):
    """Run a test module and report results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if module_name == "test_data_processor":
            from src.test.test_data_processor import test_head_trimming_logic, test_data_processor
            test_head_trimming_logic()
            test_data_processor()
        elif module_name == "test_dataset_sampler":
            from src.test.test_dataset_sampler import test_dataset, test_sampler, test_hard_negatives, test_multi_gpu
            try:
                import torchaudio
                test_dataset()
                test_sampler()
                test_hard_negatives()
                test_multi_gpu()
            except ImportError:
                print("Warning: torchaudio not available, skipping audio tests")
                return False
        elif module_name == "test_integration":
            from src.test.test_integration import test_complete_pipeline, test_flexible_clip_length
            try:
                import torchaudio
                test_complete_pipeline()
                test_flexible_clip_length()
            except ImportError:
                print("Warning: torchaudio not available, skipping integration tests")
                return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ {description} PASSED ({elapsed:.2f}s)")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} FAILED ({elapsed:.2f}s)")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    required = ["pandas", "numpy", "yaml", "torch"]
    optional = ["torchaudio"]
    
    missing_required = []
    missing_optional = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            missing_required.append(pkg)
            print(f"✗ {pkg} (required)")
    
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            missing_optional.append(pkg)
            print(f"✗ {pkg} (optional)")
    
    if missing_required:
        print(f"\nError: Missing required dependencies: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nWarning: Missing optional dependencies: {missing_optional}")
        print("Some tests will be skipped. Install with: pip install " + " ".join(missing_optional))
    
    return True


def main():
    """Run all tests."""
    print("AudioSet Data Pipeline Test Suite")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Define test modules
    tests = [
        ("test_data_processor", "Data Processor Tests"),
        ("test_dataset_sampler", "Dataset & Sampler Tests"),
        ("test_integration", "Integration Tests"),
    ]
    
    # Run tests
    results = []
    total_start = time.time()
    
    for module_name, description in tests:
        success = run_test_module(module_name, description)
        results.append((description, success))
    
    total_elapsed = time.time() - total_start
    
    # Report summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")
    
    print(f"\nResults: {passed}/{total} test suites passed")
    print(f"Total time: {total_elapsed:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 {total - passed} test suite(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
