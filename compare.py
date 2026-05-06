import torch
from pathlib import Path

base = Path("./wanda/").expanduser()

# 生成 out_0 到 out_31 的序号
out_indices = range(32)  # 0-31

print("=" * 80)
print(f"{'Layer':<12} {'Diff Ratio':<12} {'Diff Count/Total':<20} {'Status'}")
print("-" * 80)

for idx in out_indices:
    layer_name = f"self_attn.out_proj_{idx}"
    
    # 尝试多种可能的文件命名
    before_path = base / f"{layer_name}_before_mask.pt"
    after_path = base / f"{layer_name}_after_mask.pt"
    
    # 如果找不到带 _mask 的，尝试不带后缀
    if not before_path.exists():
        before_path = base / f"{layer_name}_before.pt"
    if not after_path.exists():
        after_path = base / f"{layer_name}_after.pt"
    
    if before_path.exists() and after_path.exists():
        try:
            # 加载文件
            before = torch.load(before_path, map_location='cpu')
            after = torch.load(after_path, map_location='cpu')
            
            # 处理 dict 格式
            if isinstance(before, dict):
                if 'mask' in before:
                    before = before['mask']
                elif 'weight' in before:
                    before = before['weight']
                else:
                    before = list(before.values())[0]
            
            if isinstance(after, dict):
                if 'mask' in after:
                    after = after['mask']
                elif 'weight' in after:
                    after = after['weight']
                else:
                    after = list(after.values())[0]
            
            # 检查形状
            if before.shape != after.shape:
                print(f"{layer_name:<12} {'SHAPE MISMATCH':<12} "
                      f"{str(before.shape)} vs {str(after.shape):<20}")
                continue
            
            # 计算不同元素的比例
            t1 = before.float().flatten()
            t2 = after.float().flatten()
            
            diff_count = torch.sum(t1 != t2).item()
            total_count = t1.numel()
            diff_ratio = diff_count / total_count
            diff_percentage = diff_ratio * 100
            
            # 判断状态
            if diff_ratio == 0:
                status = "✓ IDENTICAL"
            elif diff_ratio < 0.01:
                status = "○ MINOR (<1%)"
            elif diff_ratio < 0.1:
                status = "△ SOME (<10%)"
            else:
                status = f"✗ MAJOR ({diff_percentage:.1f}%)"
            
            print(f"{layer_name:<12} {diff_ratio:>10.4f}   "
                  f"{diff_count:>8}/{total_count:<8}   {status}")
            
        except Exception as e:
            print(f"{layer_name:<12} ERROR: {str(e)[:40]}")
    else:
        print(f"{layer_name:<12} {'FILE NOT FOUND':<12}")
        
print("=" * 80)