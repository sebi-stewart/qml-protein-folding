#!/usr/bin/env python3
"""
Track word count changes in a LaTeX document between writing sessions.
Uses texcount to extract word counts and maintains history in a JSON file.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class WordCountTracker:
    def __init__(self, tex_file: str, history_file: str = ".wordcount_history.json"):
        """
        Initialize the word count tracker.
        
        Args:
            tex_file: Path to the LaTeX document to track
            history_file: Path to store word count history (default: .wordcount_history.json)
        """
        self.tex_file = Path(tex_file)
        self.history_file = Path(history_file)
        
        if not self.tex_file.exists():
            raise FileNotFoundError(f"LaTeX file not found: {self.tex_file}")
        
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load word count history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not read history file {self.history_file}, starting fresh")
                return {}
        return {}
    
    def _save_history(self) -> None:
        """Save word count history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _run_texcount(self) -> str:
        """
        Run texcount on the LaTeX file and return the output.
        
        Returns:
            texcount output as a string
        """
        try:
            result = subprocess.run(
                ['texcount', str(self.tex_file)],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0 and "Command not found" in result.stderr:
                raise RuntimeError(
                    "texcount not found. Please install it:\n"
                    "  macOS: brew install texcount\n"
                    "  Linux: apt-get install texcount\n"
                    "  Or download from: https://app.uio.no/ifi/texcount/"
                )
            return result.stdout
        except FileNotFoundError:
            raise RuntimeError(
                "texcount not found. Please install it:\n"
                "  macOS: brew install texcount\n"
                "  Linux: apt-get install texcount\n"
                "  Or download from: https://app.uio.no/ifi/texcount/"
            )
    
    def _parse_texcount_output(self, output: str) -> Dict[str, int]:
        """
        Parse texcount output to extract word counts per section/subsection.
        
        Handles format from texcount with subcounts:
          text+headers+captions (#headers/#floats/#inlines/#displayed) [section_name]
        
        Args:
            output: texcount output string
            
        Returns:
            Dictionary mapping section names to word counts (text count only)
        """
        counts = {}
        lines = output.strip().split('\n')
        in_subcounts = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and separators
            if not line or line.startswith('+'):
                continue
            
            # Detect the Subcounts section
            if line.startswith('Subcounts:'):
                in_subcounts = True
                continue
            
            # Skip header line in Subcounts section
            if in_subcounts and 'text+headers+captions' in line:
                continue
            
            # Parse subcounts format: "text+headers+captions (metadata) section_name"
            # Example: "6+13+0 (1/0/0/0) _top_"
            if in_subcounts and line and not line.startswith('Subcounts'):
                # Format: NUMBER+NUMBER+NUMBER (metadata) NAME
                parts = line.split()
                if len(parts) >= 2:
                    # Extract text count (first number before +)
                    count_part = parts[0]
                    if '+' in count_part:
                        try:
                            text_count = int(count_part.split('+')[0])
                            # Section name is everything after the metadata in parentheses
                            # Find where metadata ends (last closing parenthesis)
                            paren_end = line.rfind(')')
                            if paren_end != -1:
                                section_name = line[paren_end + 1:].strip()
                                if section_name:
                                    counts[section_name] = text_count
                        except (ValueError, IndexError):
                            continue
        
        return counts
    
    def get_current_counts(self) -> Dict[str, int]:
        """Get current word counts from the LaTeX file."""
        output = self._run_texcount()
        return self._parse_texcount_output(output)
    
    def compare(self) -> None:
        """Run comparison and display results."""
        current_counts = self.get_current_counts()
        
        if not current_counts:
            print("Warning: No word counts detected. Check your LaTeX file format.")
            return
        
        # Get previous counts (most recent entry)
        previous_counts = {}
        previous_timestamp = None
        if self.history:
            # ISO format timestamps sort correctly as strings
            latest_key = max(self.history.keys())
            previous_timestamp = latest_key
            previous_counts = self.history[latest_key].get('counts', {})
        
        # Display timestamp info
        print("\n" + "=" * 110)
        print(f"Word Count Comparison for: {self.tex_file.name}")
        print("=" * 110)
        
        if previous_timestamp:
            print(f"Previous session: {previous_timestamp}")
        else:
            print("Previous session: None (first run)")
        
        print(f"Current session:  {datetime.now().isoformat()}")
        print()
        
        # Calculate and display changes
        all_sections = set(current_counts.keys()) | set(previous_counts.keys())
        
        total_previous = sum(previous_counts.values())
        total_current = sum(current_counts.values())
        total_change = total_current - total_previous
        
        print("TOTAL WORD COUNT:")
        print(f"  Previous:  {total_previous:>10,} words")
        print(f"  Current:   {total_current:>10,} words")
        print(f"  Change:    {total_change:>+10,} words ({self._percent_change(total_previous, total_current)}%)")
        print()
        
        if all_sections:
            print("SECTION-BY-SECTION CHANGES:")
            print()
            
            # Prepare data for table formatting
            rows = []
            for section in sorted(all_sections):
                prev_count = previous_counts.get(section, 0)
                curr_count = current_counts.get(section, 0)
                change = curr_count - prev_count
                
                if section not in previous_counts:
                    status = "NEW"
                elif section not in current_counts:
                    status = "REMOVED"
                else:
                    status = ""
                
                pct = self._percent_change(prev_count, curr_count)
                rows.append({
                    'section': section,
                    'prev': prev_count,
                    'curr': curr_count,
                    'change': change,
                    'pct': pct,
                    'status': status
                })
            
            # Print table header
            print(f"{'Section':<60} {'Previous':>10} {'Current':>10} {'Change':>10} {'%':>8}  Status")
            print("-" * 110)
            
            # Print table rows
            for row in rows:
                section = row['section']
                # Truncate section name if too long
                if len(section) > 49:
                    section = section[:46] + "..."
                
                status_text = f"[{row['status']}]" if row['status'] else ""
                print(f"{section:<60} {row['prev']:>10,} {row['curr']:>10,} {row['change']:>+10,} {row['pct']:>7}% {status_text}")
        
        print("=" * 110 + "\n")
        
        # Save current counts to history
        timestamp = datetime.now().isoformat()
        self.history[timestamp] = {
            'counts': current_counts,
            'total': total_current
        }
        self._save_history()
    
    @staticmethod
    def _percent_change(previous: int, current: int) -> str:
        """Calculate percent change, handling division by zero."""
        if previous == 0:
            return "∞" if current > 0 else "0"
        return f"{((current - previous) / previous * 110):.1f}"


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default to dissertation.tex if no argument provided
        tex_file = Path(__file__).parent.parent / "Dissertation_Writing" / "dissertation.tex"
    else:
        tex_file = Path(sys.argv[1])
    
    try:
        tracker = WordCountTracker(str(tex_file))
        tracker.compare()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
