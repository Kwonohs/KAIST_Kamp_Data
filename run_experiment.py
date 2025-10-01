#!/usr/bin/env python3
"""
NNI 실험 실행 스크립트
이 스크립트는 LSTM Autoencoder 이상 탐지 모델의 하이퍼파라미터 최적화를 실행합니다.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """필요한 패키지들이 설치되어 있는지 확인"""
    try:
        import nni
        import pandas
        import numpy
        import sklearn
        import tensorflow
        import matplotlib
        import seaborn
        print("✓ 모든 필요한 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"✗ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False

def check_data_files():
    """데이터 파일들이 존재하는지 확인"""
    data_files = [
        'dataset/outlier_data.csv',
        'dataset/press_data_normal.csv'
    ]
    
    missing_files = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ 다음 데이터 파일들이 없습니다:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ 모든 데이터 파일이 존재합니다.")
    return True

def run_nni_experiment():
    """NNI 실험 실행"""
    print("NNI 실험을 시작합니다...")
    print("실험 모니터링: http://localhost:8080")
    print("실험을 중단하려면 Ctrl+C를 누르세요.")
    print("-" * 50)
    
    try:
        # NNI 실험 시작
        result = subprocess.run([
            'nnictl', 'create', '--config', 'config.yml'
        ], check=True, capture_output=True, text=True)
        
        print("실험 ID:", result.stdout.strip())
        
        # 실험 상태 모니터링
        while True:
            try:
                status_result = subprocess.run([
                    'nnictl', 'experiment', 'list'
                ], capture_output=True, text=True)
                
                if 'RUNNING' in status_result.stdout:
                    print("실험이 실행 중입니다...")
                elif 'DONE' in status_result.stdout:
                    print("실험이 완료되었습니다!")
                    break
                elif 'STOPPED' in status_result.stdout:
                    print("실험이 중단되었습니다.")
                    break
                
                time.sleep(30)  # 30초마다 상태 확인
                
            except KeyboardInterrupt:
                print("\n실험을 중단합니다...")
                subprocess.run(['nnictl', 'stop'])
                break
                
    except subprocess.CalledProcessError as e:
        print(f"NNI 실험 실행 중 오류가 발생했습니다: {e}")
        print("NNI가 설치되어 있는지 확인하세요: pip install nni")
        return False
    except FileNotFoundError:
        print("nnictl 명령어를 찾을 수 없습니다.")
        print("NNI가 올바르게 설치되었는지 확인하세요.")
        return False
    
    return True

def show_results():
    """실험 결과 표시"""
    print("\n" + "="*50)
    print("실험 결과 요약")
    print("="*50)
    
    try:
        # 최고 성능 실험 정보 가져오기
        result = subprocess.run([
            'nnictl', 'experiment', 'list'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        print("\n자세한 결과는 NNI Web UI에서 확인할 수 있습니다:")
        print("http://localhost:8080")
        
    except Exception as e:
        print(f"결과를 가져오는 중 오류가 발생했습니다: {e}")

def main():
    """메인 함수"""
    print("LSTM Autoencoder 이상 탐지 - NNI 하이퍼파라미터 최적화")
    print("="*60)
    
    # 환경 확인
    if not check_requirements():
        sys.exit(1)
    
    if not check_data_files():
        print("\n데이터 파일을 준비한 후 다시 실행하세요.")
        sys.exit(1)
    
    # NNI 실험 실행
    if run_nni_experiment():
        show_results()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
