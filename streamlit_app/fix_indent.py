#!/usr/bin/env python3
# Fix indentation in NJOY Processing page

with open('pages/3_🔧_NJOY_Processing.py', 'r') as f:
    lines = f.readlines()

fixed_lines = []
fix_mode = False

for i, line in enumerate(lines):
    # Start fixing from line 129 (index 128) where "# Temperature input mode" appears
    if i == 130 and '# Temperature input mode' in line:
        fix_mode = True
    
    # Stop fixing before the "# Footer" line
    if '# Footer' in line and i > 500:
        fix_mode = False
    
    if fix_mode and line.startswith('        '):
        # Remove 4 spaces of indentation
        fixed_lines.append(line[4:])
    else:
        fixed_lines.append(line)

with open('pages/3_🔧_NJOY_Processing.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation in NJOY Processing page")
