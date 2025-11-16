#!/usr/bin/env python3
"""
Vessel Image Generation Control Script
Usage: python generate_data.py --num 500
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Vessel Image Generation Control Script')
    parser.add_argument('--num', type=int, required=True, help='Number of images to generate')
    args = parser.parse_args()
    
    # Get current script directory (Capro/scripts/)
    current_dir = Path(__file__).parent
    # Get L-System directory (Capro/data/L-System/)
    lsystem_dir = current_dir.parent / 'data' / 'L-System'
    
    # Check if L-System directory exists
    if not lsystem_dir.exists():
        print(f"Error: L-System directory not found at {lsystem_dir}")
        return 1
    
    # Switch to L-System directory
    original_cwd = os.getcwd()
    os.chdir(lsystem_dir)
    
    try:
        print(f"Generating {args.num} vessel images...")
        
        # Run make_fakevessel.py
        result = subprocess.run([
            sys.executable, 'make_fakevessel.py', 
            '--num', str(args.num)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error generating images: {result.stderr}")
            return 1
        
        print("Image generation completed, starting annotation conversion...")
        
        # Check if convert_gt.py exists
        if not os.path.exists('convert_gt.py'):
            print("Error: convert_gt.py not found in L-System directory")
            return 1
        
        # Run convert_gt.py
        result = subprocess.run([
            sys.executable, 'convert_gt.py'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting annotations: {result.stderr}")
            return 1
        
        print("Annotation conversion completed, running FDA processing...")
        
        # FDA directory is relative to current L-System directory
        fda_dir = Path('FDA')
        fda_script = fda_dir / 'FDA_retinal.py'
        
        if not fda_script.exists():
            print(f"Error: FDA_retinal.py not found at {fda_script}")
            return 1
        
        # Store the current L-System directory for later use
        current_lsystem_dir = os.getcwd()
        
        # Switch to FDA directory for running the script
        os.chdir(fda_dir)
        
        # Run FDA_retinal.py
        print("Running FDA_retinal.py...")
        result = subprocess.run([
            sys.executable, 'FDA_retinal.py'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running FDA processing: {result.stderr}")
            if result.stdout:
                print(f"FDA script output: {result.stdout}")
            return 1
        
        print("FDA processing completed...")
        
        # Switch back to L-System directory for cleanup
        os.chdir(current_lsystem_dir)
        
        print("Cleaning up temporary files...")
        
        # Delete temporary folders
        temp_folders = ['fake_rgbvessel_thin', 'fake_gtvessel_thin', 'txt_data']
        
        for folder in temp_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"Warning: Could not delete {folder}: {e}")
        
        print("Running RoLabelImg_Transform scripts...")
        
        # 退回到 data 目录
        os.chdir('..')
        
        # 检查 RoLabelImg_Transform 目录是否存在
        rolabel_dir = Path('RoLabelImg_Transform')
        
        if not rolabel_dir.exists():
            print(f"Error: RoLabelImg_Transform directory not found at {rolabel_dir.absolute()}")
            return 1
        
        # 检查脚本是否存在
        get_list_script = rolabel_dir / 'get_list.py'
        txt_to_xml_script = rolabel_dir / 'txt_to_xml.py'
        
        if not get_list_script.exists():
            print(f"Error: get_list.py not found at {get_list_script}")
            return 1
        
        if not txt_to_xml_script.exists():
            print(f"Error: txt_to_xml.py not found at {txt_to_xml_script}")
            return 1
        
        # Run get_list.py
        print("Running get_list.py...")
        result = subprocess.run([
            sys.executable, 'RoLabelImg_Transform/get_list.py'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running get_list.py: {result.stderr}")
            if result.stdout:
                print(f"get_list.py output: {result.stdout}")
            return 1
        
        # Run txt_to_xml.py
        print("Running txt_to_xml.py...")
        result = subprocess.run([
            sys.executable, 'RoLabelImg_Transform/txt_to_xml.py'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running txt_to_xml.py: {result.stderr}")
            if result.stdout:
                print(f"txt_to_xml.py output: {result.stdout}")
            return 1
        
        print("RoLabelImg_Transform processing completed!")
        # 新增：运行 voc2coco.py
        print("Running voc2coco.py conversion...")
        
        # 检查 detector/labelGenerator 目录是否存在
        detector_dir = Path('..') / 'detector' / 'labelGenerator'
        voc2coco_script = detector_dir / 'voc2coco.py'
        
        if not voc2coco_script.exists():
            print(f"Error: voc2coco.py not found at {voc2coco_script}")
            return 1
        
        # 运行 voc2coco.py
        print("Running voc2coco.py...")
        result = subprocess.run([
            sys.executable, str(voc2coco_script)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running voc2coco.py: {result.stderr}")
            if result.stdout:
                print(f"voc2coco.py output: {result.stdout}")
            return 1
        
        print("voc2coco conversion completed successfully!")
        print("All tasks completed successfully!")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Switch back to original directory
        os.chdir(original_cwd)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())