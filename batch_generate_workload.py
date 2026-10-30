#!/usr/bin/env python3

import os
import re
import argparse
import subprocess
import shutil
from pathlib import Path
import glob

def extract_latency_from_filename(filename: str) -> int:
    match = re.search(r'Latency(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从文件名 '{filename}' 中提取 Latency 数值")

def main():
    parser = argparse.ArgumentParser(description="串行运行 HBF 推理 workload，结果文件在 output_base_dir 下")
    parser.add_argument('--input_dir', required=True, help='包含多个 -a 输入文件的目录')
    parser.add_argument('--config_file', required=True, help='配置文件路径 (-c 参数)')
    parser.add_argument('--model_output_prefix', required=True,
                        help='结果文件前缀，例如 "Qwen3-Moe-235B"')
    parser.add_argument('--output_base_dir', required=True, help='输出根目录（shell 脚本会在此目录生成结果文件）')

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    config_file = Path(args.config_file).resolve()
    output_base_dir = Path(args.output_base_dir).resolve()
    output_prefix = args.model_output_prefix

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    input_files = sorted(input_dir.glob("*.txt"))
    if not input_files:
        print(f"警告：在 {input_dir} 中未找到任何 .txt 文件")
        return

    output_base_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path("./scripts/HBF_inference_workload_with_computeCache.sh").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"推理脚本未找到: {script_path}")

    # 关键：结果文件由 shell 脚本生成在 output_base_dir 下！
    result_search_dir = output_base_dir

    for file_path in input_files:
        try:
            latency = extract_latency_from_filename(file_path.name)
        except ValueError as e:
            print(f"跳过文件 {file_path.name}: {e}")
            continue

        out_subdir = output_base_dir / f"latency{latency}"
        out_subdir.mkdir(exist_ok=True)
        log_file = out_subdir / "run.log"

        cmd = [
            "sh", str(script_path),
            "-m", "qwen3-235B",
            "-c", str(config_file),
            "-a", str(file_path)
        ]

        print(f"正在运行: latency={latency} → 输出目录: {out_subdir}")
        print(f"命令: {' '.join(cmd)}")

        # 记录运行前已存在的结果文件（防干扰）
        existing_files = set(result_search_dir.glob(f"{output_prefix}-*.txt"))

        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, cwd=os.getcwd(), stdout=f, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            print(f"❌ 脚本执行失败: latency{latency}，退出码 {result.returncode}")
            print(f"请查看日志: {log_file}")
            exit(1)

        # 查找运行后新增的结果文件（在 output_base_dir 下！）
        current_files = set(result_search_dir.glob(f"{output_prefix}-*.txt"))
        new_files = current_files - existing_files

        if not new_files:
            print(f"❌ 错误: latency{latency} 运行成功，但在 {result_search_dir} 中未找到新生成的 '{output_prefix}-*.txt' 文件！")
            print(f"该目录下所有匹配文件: {[f.name for f in result_search_dir.glob('*.txt')]}")
            exit(1)

        # 移动新文件到 latency 子目录
        for src in new_files:
            dst = out_subdir / src.name
            shutil.move(str(src), str(dst))
            print(f"📁 移动结果文件: {src.name} → {out_subdir}")

        print(f"✅ 成功完成 latency{latency}，共移动 {len(new_files)} 个结果文件")

    print("🎉 所有实验完成！")

if __name__ == "__main__":
    main()