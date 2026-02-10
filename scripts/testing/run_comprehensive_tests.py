#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner with Metrics

Runs all tests and provides detailed metrics including:
- Pass/fail rates
- Code coverage
- Performance metrics
- Model accuracy metrics
- Feature-specific test results
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pytest


class TestMetricsCollector:
    """Collects and reports comprehensive test metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall": {},
            "by_category": {},
            "coverage": {},
            "performance": {},
            "failures": [],
            "warnings": []
        }
    
    def run_tests(self) -> int:
        """Run all tests with coverage and collect metrics."""
        print("=" * 80)
        print("ShikshaSetu Comprehensive Test Suite")
        print("=" * 80)
        print(f"Started: {self.results['timestamp']}")
        print()
        
        self.start_time = time.time()
        
        # Run pytest with coverage
        print("Running tests with coverage analysis...")
        print("-" * 80)
        
        pytest_args = [
            "tests/",
            "-v",
            "--cov=backend",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=json:test-results/coverage.json",
            "--cov-report=html:test-results/htmlcov",
            "--junit-xml=test-results/junit.xml",
            "--tb=short",
            "-x",  # Stop on first failure for detailed output
        ]
        
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        self.end_time = time.time()
        self.results["overall"]["duration_seconds"] = round(self.end_time - self.start_time, 2)
        
        # Parse results
        self._parse_test_results(exit_code)
        self._parse_coverage_results()
        
        # Display summary
        self._display_summary()
        
        return exit_code
    
    def _parse_test_results(self, exit_code: int):
        """Parse test results from pytest output."""
        # Check if junit xml exists
        junit_path = Path("test-results/junit.xml")
        if junit_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(junit_path)
                root = tree.getroot()
                
                # Get test suite stats
                testsuite = root.find('.//testsuite')
                if testsuite is not None:
                    self.results["overall"]["total_tests"] = int(testsuite.get('tests', 0))
                    self.results["overall"]["passed"] = int(testsuite.get('tests', 0)) - int(testsuite.get('failures', 0)) - int(testsuite.get('errors', 0))
                    self.results["overall"]["failed"] = int(testsuite.get('failures', 0))
                    self.results["overall"]["errors"] = int(testsuite.get('errors', 0))
                    self.results["overall"]["skipped"] = int(testsuite.get('skipped', 0))
                    
                    # Parse individual test cases
                    for testcase in root.findall('.//testcase'):
                        classname = testcase.get('classname', '')
                        name = testcase.get('name', '')
                        time_taken = float(testcase.get('time', 0))
                        
                        # Check for failures
                        failure = testcase.find('failure')
                        if failure is not None:
                            self.results["failures"].append({
                                "test": f"{classname}::{name}",
                                "message": failure.get('message', ''),
                                "type": failure.get('type', '')
                            })
            except Exception as e:
                print(f"Warning: Could not parse junit.xml: {e}")
        else:
            # Estimate from exit code
            self.results["overall"]["exit_code"] = exit_code
            if exit_code == 0:
                self.results["overall"]["status"] = "PASSED"
            else:
                self.results["overall"]["status"] = "FAILED"
    
    def _parse_coverage_results(self):
        """Parse coverage results from coverage.json."""
        coverage_path = Path("test-results/coverage.json")
        if coverage_path.exists():
            try:
                with open(coverage_path, 'r') as f:
                    coverage_data = json.load(f)
                
                totals = coverage_data.get('totals', {})
                self.results["coverage"] = {
                    "percent_covered": round(totals.get('percent_covered', 0), 2),
                    "lines_covered": totals.get('covered_lines', 0),
                    "lines_total": totals.get('num_statements', 0),
                    "lines_missing": totals.get('missing_lines', 0),
                    "branches_covered": totals.get('covered_branches', 0),
                    "branches_total": totals.get('num_branches', 0)
                }
                
                # Files with low coverage
                files = coverage_data.get('files', {})
                low_coverage_files = []
                for filepath, file_data in files.items():
                    summary = file_data.get('summary', {})
                    percent = summary.get('percent_covered', 0)
                    if percent < 80 and percent > 0:
                        low_coverage_files.append({
                            "file": filepath,
                            "coverage": round(percent, 2)
                        })
                
                if low_coverage_files:
                    self.results["warnings"].append({
                        "type": "low_coverage",
                        "count": len(low_coverage_files),
                        "files": sorted(low_coverage_files, key=lambda x: x['coverage'])[:5]
                    })
                    
            except Exception as e:
                print(f"Warning: Could not parse coverage.json: {e}")
    
    def _display_summary(self):
        """Display comprehensive test summary."""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        # Overall stats
        overall = self.results["overall"]
        print("\n📊 Overall Statistics:")
        print("-" * 40)
        if "total_tests" in overall:
            total = overall["total_tests"]
            passed = overall.get("passed", 0)
            failed = overall.get("failed", 0)
            errors = overall.get("errors", 0)
            skipped = overall.get("skipped", 0)
            
            print(f"  Total Tests:     {total}")
            print(f"  ✓ Passed:        {passed} ({(passed/total*100) if total > 0 else 0:.1f}%)")
            print(f"  ✗ Failed:        {failed}")
            print(f"  ⚠ Errors:        {errors}")
            print(f"  ⊝ Skipped:       {skipped}")
        print(f"  Duration:        {overall.get('duration_seconds', 0):.2f}s")
        
        # Coverage
        coverage = self.results.get("coverage", {})
        if coverage:
            print("\n📈 Code Coverage:")
            print("-" * 40)
            percent = coverage.get("percent_covered", 0)
            lines_covered = coverage.get("lines_covered", 0)
            lines_total = coverage.get("lines_total", 0)
            lines_missing = coverage.get("lines_missing", 0)
            
            # Color code coverage
            if percent >= 80:
                color = "\033[92m"  # Green
            elif percent >= 60:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            reset = "\033[0m"
            
            print(f"  Overall:         {color}{percent}%{reset}")
            print(f"  Lines Covered:   {lines_covered}/{lines_total}")
            print(f"  Lines Missing:   {lines_missing}")
            
            if coverage.get("branches_total", 0) > 0:
                branches_covered = coverage.get("branches_covered", 0)
                branches_total = coverage.get("branches_total", 0)
                branch_percent = (branches_covered / branches_total * 100) if branches_total > 0 else 0
                print(f"  Branch Coverage: {branch_percent:.1f}% ({branches_covered}/{branches_total})")
        
        # Failures
        failures = self.results.get("failures", [])
        if failures:
            print(f"\n❌ Failed Tests ({len(failures)}):")
            print("-" * 40)
            for i, failure in enumerate(failures[:5], 1):
                print(f"  {i}. {failure['test']}")
                if failure.get('message'):
                    msg = failure['message'][:100]
                    print(f"     {msg}")
            if len(failures) > 5:
                print(f"  ... and {len(failures) - 5} more")
        
        # Warnings
        warnings = self.results.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            print("-" * 40)
            for warning in warnings:
                if warning['type'] == 'low_coverage':
                    print(f"  Low coverage in {warning['count']} files:")
                    for file_info in warning['files']:
                        print(f"    - {file_info['file']}: {file_info['coverage']}%")
        
        # Final status
        print("\n" + "=" * 80)
        if overall.get("status") == "PASSED" or (overall.get("failed", 0) == 0 and overall.get("errors", 0) == 0):
            print("\033[92m✓ ALL TESTS PASSED\033[0m")
        else:
            print("\033[91m✗ SOME TESTS FAILED\033[0m")
        print("=" * 80)
        
        # Save detailed report
        self._save_report()
    
    def _save_report(self):
        """Save detailed JSON report."""
        report_path = Path("test-results/test-metrics.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Main test runner."""
    # Create test results directory
    Path("test-results").mkdir(exist_ok=True)
    
    # Run tests
    collector = TestMetricsCollector()
    exit_code = collector.run_tests()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
