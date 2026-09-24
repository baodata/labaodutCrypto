with open('configs/assets.yaml', 'r') as f:
    text = f.read()

bad_index = """  sector: "index"
    agent_group: "index"
"""
good_index = """  sector: "index"
"""

text = text.replace(bad_index, good_index)

# Also check for trailing spaces or other issues just in case
with open('configs/assets.yaml', 'w') as f:
    f.write(text)
