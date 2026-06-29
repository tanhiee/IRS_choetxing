import sys

def run():
    bt = chr(96)
    
    # 1. Clean English draft
    fn = 'ieee_paper_incident_response_rl.md'
    content = open(fn, encoding='utf-8').read()
    content = content.replace('\r\n', '\n')
    
    old_str = f'* **Variant B (Light Wasted Penalty)**: {bt}wasted_restore = -10.0{bt} (reduced from the default value of `-30.0` to evaluate sensitivity) (reduced from the default value of `-30.0` to evaluate sensitivity)'
    new_str = f'* **Variant B (Light Wasted Penalty)**: {bt}wasted_restore = -10.0{bt} (reduced from the default value of `-30.0` to evaluate sensitivity)'
    
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(content)
        print('English draft cleaned successfully.')
    else:
        print('English target not found.')

    # 2. Patch Vietnamese drafts
    vi_files = ['ieee_paper_incident_response_rl_vi.md', 'ieee_paper_incident_response_vi.md']
    for vi_fn in vi_files:
        vi_content = open(vi_fn, encoding='utf-8').read()
        vi_content = vi_content.replace('\r\n', '\n')
        
        old_vi = f'* **Biến thể B (Phạt nhẹ Restore lỗi)**: {bt}wasted_restore = -10.0{bt}'
        new_vi = f'* **Biến thể B (Phạt nhẹ Restore lỗi)**: {bt}wasted_restore = -10.0{bt} (giảm từ mức mặc định `-30.0` để đánh giá độ nhạy)'
        
        if old_vi in vi_content:
            vi_content = vi_content.replace(old_vi, new_vi)
            with open(vi_fn, 'w', encoding='utf-8') as f:
                f.write(vi_content)
            print(f'{vi_fn} patched successfully.')
        else:
            if new_vi in vi_content:
                print(f'{vi_fn} was already patched.')
            else:
                print(f'{vi_fn} target not found.')

if __name__ == '__main__':
    run()
