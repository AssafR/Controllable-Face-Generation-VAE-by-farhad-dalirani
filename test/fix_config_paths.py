#!/usr/bin/env python3
"""
Fix config paths in all test scripts to work from both main and test directories.
"""

import os
import re

def fix_config_paths_in_file(file_path):
    """Fix config paths in a single file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix hardcoded config/config.json paths
    if 'config/config.json' in content:
        # Replace with flexible path logic
        old_pattern = r'with open\([\'"]config/config\.json[\'"], [\'"]r[\'"]\) as [a-zA-Z_]+:'
        new_pattern = '''# Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = "config/config.json"
    
    with open(config_path, 'r') as file:'''
        
        content = re.sub(old_pattern, new_pattern, content)
        
        # Add os import if not present
        if 'import os' not in content and 'os.path.exists' in content:
            # Find the first import and add os import
            lines = content.split('\n')
            new_lines = []
            added_os_import = False
            
            for line in lines:
                if line.startswith('import ') and not added_os_import:
                    new_lines.append('import os')
                    new_lines.append(line)
                    added_os_import = True
                else:
                    new_lines.append(line)
            
            if not added_os_import:
                # Add at the beginning if no imports found
                new_lines.insert(0, 'import os')
            
            content = '\n'.join(new_lines)
    
    # Fix glob patterns for config files
    if 'glob.glob("config/config_*.json")' in content:
        content = content.replace(
            'glob.glob("config/config_*.json")',
            'glob.glob("../config/config_*.json") if os.path.exists("../config/") else glob.glob("config/config_*.json")'
        )
    
    # Fix other hardcoded config paths
    content = content.replace('"config/config.json"', 'config_path')
    content = content.replace("'config/config.json'", 'config_path')
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated: {file_path}")
        return True
    else:
        print(f"⏭️  No changes needed: {file_path}")
        return False

def main():
    """Fix config paths in all Python files."""
    
    print("🔧 Fixing config paths in test files...")
    print("=" * 50)
    
    # Get all Python files in current directory
    python_files = [f for f in os.listdir('.') if f.endswith('.py') and f not in ['fix_config_paths.py', 'update_imports.py']]
    
    updated_count = 0
    for file_path in python_files:
        try:
            if fix_config_paths_in_file(file_path):
                updated_count += 1
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n✅ Updated {updated_count} files!")
    print("Now all test scripts should work from both main and test directories.")

if __name__ == "__main__":
    main()
