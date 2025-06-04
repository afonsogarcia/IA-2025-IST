#!/usr/bin/env python3
"""
Simple test runner for Nuruomino project
Runs all test cases from both sample-nuruominoboards and public folders and compares results
"""

import os
import subprocess
import glob
import time

def run_single_test(test_name):
    """Run a single test and compare with expected output"""
    # Try to find test input in both locations
    test_input_sample = f"..\\sample-nuruominoboards\\{test_name}.txt"
    test_input_public = f"..\\..\\public\\{test_name}.txt"
    
    test_input = None
    if os.path.exists(test_input_sample):
        test_input = test_input_sample
        expected_output_txt = f"..\\sample-nuruominoboards\\{test_name}.out.txt"
        expected_output = f"..\\sample-nuruominoboards\\{test_name}.out"
    elif os.path.exists(test_input_public):
        test_input = test_input_public
        expected_output_txt = f"..\\..\\public\\{test_name}.out.txt"
        expected_output = f"..\\..\\public\\{test_name}.out"
    else:
        print(f"❌ Input file not found: {test_name}.txt")
        return False
    
    print(f"Testing {test_name}... ", end="", flush=True)
      # Try both .out.txt and .out extensions for expected output
    # Also try without dash for test03 case
    expected_file = None
    possible_outputs = [
        expected_output_txt,  # test-XX.out.txt
        expected_output,      # test-XX.out
    ]
    
    # Add alternative naming for inconsistent file names (like test03.out.txt instead of test-03.out.txt)
    if test_input == test_input_sample:
        alt_name = test_name.replace('-', '')
        possible_outputs.extend([
            f"..\\sample-nuruominoboards\\{alt_name}.out.txt",
            f"..\\sample-nuruominoboards\\{alt_name}.out"
        ])
    elif test_input == test_input_public:
        alt_name = test_name.replace('-', '')
        possible_outputs.extend([
            f"..\\public\\{alt_name}.out.txt",
            f"..\\public\\{alt_name}.out"
        ])
    
    for output_file in possible_outputs:
        if os.path.exists(output_file):
            expected_file = output_file
            break
    
    if expected_file is None:
        print(f"❌ Expected output file not found for {test_name}")
        return False
    
    try:
        # Run the test
        start_time = time.time()
        cmd = f'Get-Content "{test_input}" | python nuruomino.py'
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=30
        )
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"❌ Error running test: {result.stderr}")
            return False
        
        # Read expected output
        with open(expected_file, 'r') as f:
            expected = f.read().strip()
        
        actual = result.stdout.strip()
        
        # Compare (normalize whitespace)
        def normalize(text):
            return '\n'.join('\t'.join(line.split()) for line in text.split('\n') if line.strip())
        
        if normalize(actual) == normalize(expected):
            print(f"✅ PASSED ({end_time - start_time:.2f}s)")
            return True
        else:
            print(f"❌ FAILED ({end_time - start_time:.2f}s)")
            print(f"Expected first line: {expected.split('\n')[0] if expected else '(empty)'}")
            print(f"Actual first line:   {actual.split('\n')[0] if actual else '(empty)'}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("Nuruomino Test Runner")
    print("=" * 50)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
      # Find all test files in both locations
    sample_dir = "../sample-nuruominoboards"
    public_dir = "../../public"
    
    test_files = []
    test_files.extend(glob.glob(f"{sample_dir}/test*.txt"))
    test_files.extend(glob.glob(f"{public_dir}/test*.txt"))
    
    if not test_files:
        print("No test files found!")
        return
    
    # Extract test names (remove path and extension), exclude .out.txt files
    test_names = []
    for f in test_files:
        filename = os.path.basename(f)
        if not filename.endswith('.out.txt'):  # Skip output files
            test_names.append(filename[:-4])  # Remove .txt
    test_names = list(set(test_names))  # Remove duplicates
    test_names.sort()
    
    print(f"Found {len(test_names)} tests: {', '.join(test_names)}")
    print("-" * 50)
    
    # Run all tests
    passed = 0
    total = len(test_names)
    
    for test_name in test_names:
        if run_single_test(test_name):
            passed += 1
    
    print("-" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"💥 {total - passed} tests failed")

if __name__ == "__main__":
    main()
