#!/usr/bin/env python3
"""
Progress tracking utility for model downloads.
Provides visual progress bars and download statistics.
"""

import sys
import time
from pathlib import Path


class ProgressBar:
    """Visual progress bar with percentage and statistics."""
    
    def __init__(self, total_items, width=40, name="Progress"):
        self.total = total_items
        self.current = 0
        self.width = width
        self.name = name
        self.start_time = time.time()
        self.completed_items = []
        self.failed_items = []
    
    def update(self, item_name, status="downloading"):
        """Update progress for current item."""
        self.current += 1
        pct = (self.current * 100) // self.total
        elapsed = time.time() - self.start_time
        
        # Calculate bar
        filled = (pct * self.width) // 100
        empty = self.width - filled
        bar = "▓" * filled + "░" * empty
        
        # Calculate ETA
        if self.current > 0 and elapsed > 0:
            avg_time = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = int(avg_time * remaining_items)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Print progress line
        print(f"\r  [{pct:3d}%] ▶ {bar} ◀ ({self.current}/{self.total}) ETA: {eta_str}", end="", flush=True)
    
    def complete_item(self, item_name, success=True):
        """Mark item as complete."""
        if success:
            self.completed_items.append(item_name)
            status = "✓"
            color = "\033[0;32m"  # Green
        else:
            self.failed_items.append(item_name)
            status = "⚠"
            color = "\033[1;33m"  # Yellow
        
        # Clear current line and show result
        print()  # New line
        print(f"     {color}{status}\033[0m {item_name}")
    
    def finish(self):
        """Print final summary."""
        print()
        elapsed = time.time() - self.start_time
        
        # Summary bar
        success_pct = (len(self.completed_items) * 100) // self.total if self.total > 0 else 0
        filled = (success_pct * self.width) // 100
        empty = self.width - filled
        bar = "▓" * filled + "░" * empty
        
        print(f"\n  ════════════════════════════════════════════════════════")
        print(f"  Summary: {bar}")
        print(f"  Completed: \033[0;32m{len(self.completed_items)}/{self.total}\033[0m ({success_pct}%)")
        if self.failed_items:
            print(f"  Failed: \033[1;33m{len(self.failed_items)}\033[0m (will auto-download on first use)")
        print(f"  Time elapsed: {self._format_time(int(elapsed))}")
        print(f"  ════════════════════════════════════════════════════════")
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}m {secs}s"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours}h {mins}m"


class ModelDownloadTracker:
    """Track model downloads with detailed statistics."""
    
    def __init__(self, models_list):
        self.models = models_list
        self.total = len(models_list)
        self.progress = ProgressBar(self.total, name="Model Download")
        self.downloaded_size = 0
        self.total_size = 0
    
    def start_download(self, model_name, model_size):
        """Start downloading a model."""
        print(f"\n  ┌─ {model_name} ({model_size})")
        self.progress.update(model_name, "downloading")
    
    def log_step(self, step_msg):
        """Log a download step."""
        print(f"\n     → {step_msg}", end="", flush=True)
    
    def complete_download(self, model_name, success=True):
        """Mark model download as complete."""
        print()  # New line after step
        self.progress.complete_item(model_name, success)
    
    def finish_all(self):
        """Finish all downloads and show summary."""
        self.progress.finish()


def print_model_info(model_name, model_id, model_size, description):
    """Print formatted model information."""
    print(f"\n  ╭─ {model_name}")
    print(f"  │  ID: {model_id}")
    print(f"  │  Purpose: {description}")
    print(f"  ╰─ Size: {model_size}")


if __name__ == "__main__":
    # Demo
    models = [
        {"name": "Qwen3-8B", "size": "4.6GB"},
        {"name": "IndicTrans2", "size": "2GB"},
        {"name": "BGE-M3", "size": "1.2GB"},
        {"name": "BGE-Reranker", "size": "1GB"}
    ]
    
    tracker = ModelDownloadTracker(models)
    
    for model in models:
        tracker.start_download(model["name"], model["size"])
        time.sleep(1)
        tracker.log_step("Downloading files...")
        time.sleep(1)
        tracker.complete_download(model["name"], True)
    
    tracker.finish_all()
