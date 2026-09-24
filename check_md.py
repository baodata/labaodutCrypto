import re

def check_table(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    in_table = False
    col_count = -1
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            # Replace \| with a dummy character so it doesn't split
            clean_line = line.replace('\\|', 'PIPE')
            cols = clean_line.split('|')
            if col_count == -1:
                col_count = len(cols)
            elif len(cols) != col_count:
                print(f"Error in table {filename} at line {i+1}: expected {col_count} cols, got {len(cols)}")
                return False
        else:
            col_count = -1
    print(f"All tables in {filename} are well-formatted!")
    return True

check_table('manual/BANG_DIEU_PHOI_2_NGUOI.md')
