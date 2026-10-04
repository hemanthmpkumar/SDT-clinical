import json

gdt_code = None
spd_code = None

with open('/Users/hemanth/.gemini/antigravity/brain/14ab0738-6c13-4b33-b59e-4942c8f1670b/.system_generated/logs/transcript_full.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'GENERIC':
            content = data.get('content', '')
            if 'File Path: `file:///Users/hemanth/Projects/Papers/SDT/src/models/gdt.py`' in content:
                gdt_code = content
            elif 'File Path: `file:///Users/hemanth/Projects/Papers/SDT/src/models/spd_encoder.py`' in content:
                spd_code = content

def clean_code(raw_content):
    if not raw_content: return ""
    lines = raw_content.split('\n')
    out = []
    start_collecting = False
    for line in lines:
        if line.startswith('1: '):
            start_collecting = True
        if start_collecting and 'The above content shows the entire' in line:
            break
        if start_collecting:
            # remove line numbers like "123: "
            idx = line.find(': ')
            if idx != -1 and line[:idx].isdigit():
                out.append(line[idx+2:])
            else:
                out.append(line)
    return '\n'.join(out)

if gdt_code:
    with open('src/models/gdt.py', 'w') as f:
        f.write(clean_code(gdt_code))
    print("Recovered gdt.py")

if spd_code:
    with open('src/models/spd_encoder.py', 'w') as f:
        f.write(clean_code(spd_code))
    print("Recovered spd_encoder.py")
