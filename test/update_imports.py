#!/usr/bin/env python3
"""
Update import paths in test files to work from the test/ directory.
"""

import os
import re

def update_file_imports(file_path):
    """Update imports in a single file to work from test/ directory."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add sys.path modification at the top if not already present
    if 'sys.path' not in content and ('from ' in content or 'import ' in content):
        # Find the first import and add sys.path before it
        lines = content.split('\n')
        new_lines = []
        
        # Add sys.path modification
        new_lines.append('import sys')
        new_lines.append('import os')
        new_lines.append('# Add parent directory to path for imports')
        new_lines.append('sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
        new_lines.append('')
        
        # Add all original lines
        new_lines.extend(lines)
        
        content = '\n'.join(new_lines)
    
    # Write back the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated: {file_path}")

def main():
    """Update all Python files in the test directory."""
    
    print("🔧 Updating import paths in test files...")
    print("=" * 50)
    
    # Get all Python files in current directory
    test_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'update_imports.py']
    
    for file_path in test_files:
        try:
            update_file_imports(file_path)
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n✅ Updated {len(test_files)} files!")
    print("Now you can run test files from the test/ directory with:")
    print("uv run test/filename.py")

if __name__ == "__main__":
    main()
