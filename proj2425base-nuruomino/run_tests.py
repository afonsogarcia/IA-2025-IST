#!/usr/bin/env python3
"""
Advanced test runner for Nuruomino project
Runs all test cases from both sample-nuruominoboards and public folders
Uses parallel execution with clean, organized output showing test progress and results
All tests start simultaneously for maximum efficiency
"""

import os
import subprocess
import glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Lock for synchronized printing
print_lock = Lock()

class TestRunner:
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.running_tests = set()
        self.test_status = {}  # Track test status for better display
        
    def normalize_text(self, text):
        """Normalize whitespace for comparison"""
        return '\n'.join('\t'.join(line.split()) for line in text.split('\n') if line.strip())
    
    def find_test_files(self, test_name):
        """Find input and expected output files for a test"""
        # Try sample-nuruominoboards first
        test_input_sample = f"..\\sample-nuruominoboards\\{test_name}.txt"
        test_input_public = f"..\\public\\{test_name}.txt"
        
        test_input = None
        expected_outputs = []
        
        if os.path.exists(test_input_sample):
            test_input = test_input_sample
            expected_outputs = [
                f"..\\sample-nuruominoboards\\{test_name}.out.txt",
                f"..\\sample-nuruominoboards\\{test_name}.out",
                f"..\\sample-nuruominoboards\\{test_name.replace('-', '')}.out.txt",
                f"..\\sample-nuruominoboards\\{test_name.replace('-', '')}.out"
            ]
        elif os.path.exists(test_input_public):
            test_input = test_input_public
            expected_outputs = [
                f"..\\public\\{test_name}.out.txt",
                f"..\\public\\{test_name}.out",
                f"..\\public\\{test_name.replace('-', '')}.out.txt",
                f"..\\public\\{test_name.replace('-', '')}.out"
            ]
        
        # Find the first existing expected output file
        expected_file = None
        for output_file in expected_outputs:
            if os.path.exists(output_file):
                expected_file = output_file
                break
                
        return test_input, expected_file
    
    def get_timeout(self, test_name):
        """Get timeout based on test complexity"""
        complex_tests = ['test04', 'test05', 'test06', 'test07', 'test09', 'test10', 
                        'test11', 'test12', 'test13', 'test15']
        if any(complex_name in test_name for complex_name in complex_tests):
            return 90  # Extended timeout for complex tests
        return 30  # Default timeout
    
    def update_test_status(self, test_name, status, duration=None, details=None):
        """Update test status with thread-safe printing and enhanced readability"""
        with print_lock:
            self.test_status[test_name] = {
                'status': status,
                'duration': duration,
                'details': details,
                'timestamp': time.time()
            }
            
            # Print real-time update with better formatting
            current_time = time.strftime("%H:%M:%S")
            if status == "RUNNING":
                print(f"🔄 {current_time} | {test_name:12} | ▶️  INICIADO")
            elif status == "PASSED":
                print(f"✅ {current_time} | {test_name:12} | ✨ PASSOU em {duration:.2f}s")
            elif status == "FAILED":
                print(f"❌ {current_time} | {test_name:12} | ❌ FALHOU em {duration:.2f}s")
                if details:
                    print(f"   🔍 Detalhes: {details}")
            elif status == "TIMEOUT":
                print(f"⏰ {current_time} | {test_name:12} | ⏰ TIMEOUT após {duration:.0f}s")
            elif status == "ERROR":
                print(f"💥 {current_time} | {test_name:12} | 💥 ERRO")
                if details:
                    print(f"   🔍 Erro: {details}")
            elif status == "NOT_FOUND":
                print(f"📁 {current_time} | {test_name:12} | 📁 FICHEIROS NÃO ENCONTRADOS")
    
    def run_single_test(self, test_name):
        """Run a single test and return result"""
        test_input, expected_file = self.find_test_files(test_name)
        
        if not test_input or not expected_file:
            self.update_test_status(test_name, "NOT_FOUND")
            return False
        
        self.update_test_status(test_name, "RUNNING")
        timeout = self.get_timeout(test_name)
        
        try:
            start_time = time.time()
            cmd = f'Get-Content "{test_input}" | python nuruomino.py'
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode != 0:
                error_msg = result.stderr.strip()[:100] if result.stderr else "Erro desconhecido"
                self.update_test_status(test_name, "ERROR", duration, error_msg)
                return False
            
            # Read expected output
            with open(expected_file, 'r') as f:
                expected = f.read().strip()
            
            actual = result.stdout.strip()
            
            # Compare normalized output
            if self.normalize_text(actual) == self.normalize_text(expected):
                self.update_test_status(test_name, "PASSED", duration)
                return True
            else:
                # Show first difference for debugging
                expected_lines = expected.split('\n')
                actual_lines = actual.split('\n')
                detail = f"Esperado: '{expected_lines[0][:40]}...', Obtido: '{actual_lines[0][:40]}...'"
                self.update_test_status(test_name, "FAILED", duration, detail)
                return False
                
        except subprocess.TimeoutExpired:
            self.update_test_status(test_name, "TIMEOUT", timeout)
            return False
        except Exception as e:
            self.update_test_status(test_name, "ERROR", 0, str(e)[:100])
            return False
    
    def collect_test_names(self):
        """Collect all available test names"""
        sample_dir = "../sample-nuruominoboards"
        public_dir = "../public"
        
        test_files = []
        if os.path.exists(sample_dir):
            test_files.extend(glob.glob(f"{sample_dir}/test*.txt"))
        if os.path.exists(public_dir):
            test_files.extend(glob.glob(f"{public_dir}/test*.txt"))
        
        if not test_files:
            return []
        
        # Extract test names, exclude output files
        test_names = []
        for f in test_files:
            filename = os.path.basename(f)
            if not filename.endswith('.out.txt') and filename.endswith('.txt'):
                test_names.append(filename[:-4])  # Remove .txt
        
        return sorted(list(set(test_names)))  # Remove duplicates and sort
    
    def print_header(self, test_names):
        """Print formatted header"""
        print()
        print("🧩 EXECUTOR DE TESTES NURUOMINO - EXECUÇÃO SIMULTÂNEA TOTAL")
        print("=" * 80)
        print(f"📁 Encontrados {len(test_names)} testes: {', '.join(test_names)}")
        
        # Calculate max workers
        max_workers = min(os.cpu_count() or 4, len(test_names))
        print(f"⚡ Usando {max_workers} processos paralelos para execução máxima")
        print(f"🚀 TODOS OS TESTES COMEÇAM SIMULTANEAMENTE!")
        print("=" * 80)
        print()
    
    def print_live_status(self, total_tests):
        """Print live status every few seconds with enhanced display"""
        last_update_time = time.time()
        
        while len(self.results) < total_tests:
            time.sleep(3)  # Update every 3 seconds
            current_time = time.time()
            
            with print_lock:
                completed = len(self.results)
                running_tests = [t for t, s in self.test_status.items() if s.get('status') == 'RUNNING']
                pending_tests = [t for t, s in self.test_status.items() if s.get('status') == 'QUEUED']
                running = len(running_tests)
                queued = len(pending_tests)
                progress = (completed / total_tests) * 100 if total_tests > 0 else 0
                elapsed = current_time - self.start_time
                
                # Only print if there's been a reasonable time gap
                if current_time - last_update_time >= 2.5:
                    print(f"📊 [{time.strftime('%H:%M:%S')}] Progresso: {completed}/{total_tests} ({progress:.1f}%) | "
                          f"Executando: {running} | Pendentes: {queued} | Tempo: {elapsed:.1f}s")
                    
                    # Show which tests are still running and pending
                    if running_tests:
                        running_list = ', '.join(sorted(running_tests))
                        print(f"   🔄 Executando: {running_list}")
                    
                    if pending_tests and len(pending_tests) <= 10:  # Only show if not too many
                        pending_list = ', '.join(sorted(pending_tests))
                        print(f"   ⏳ Pendentes: {pending_list}")
                    elif pending_tests:
                        print(f"   ⏳ Pendentes: {len(pending_tests)} testes restantes")
                    
                    last_update_time = current_time
    
    def print_summary(self, total_duration):
        """Print final summary with enhanced readability"""
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        print()
        print("=" * 80)
        print(f"📊 RESULTADOS FINAIS: {passed}/{total} testes passaram")
        print(f"⏱️  Tempo total: {total_duration:.1f}s")
        print("=" * 80)
        
        # Separate passed and failed tests
        passed_tests = sorted([test for test, result in self.results.items() if result])
        failed_tests = sorted([test for test, result in self.results.items() if not result])
        
        if passed_tests:
            print(f"✅ TESTES QUE PASSARAM ({len(passed_tests)}):")
            for i, test in enumerate(passed_tests, 1):
                status = self.test_status.get(test, {})
                duration = status.get('duration', 0)
                print(f"   {i:2d}. {test:12} ({duration:.2f}s)")
        
        if failed_tests:
            print(f"\n❌ TESTES QUE FALHARAM ({len(failed_tests)}):")
            for i, test in enumerate(failed_tests, 1):
                status = self.test_status.get(test, {})
                duration = status.get('duration', 0)
                details = status.get('details', '')
                print(f"   {i:2d}. {test:12} ({duration:.2f}s)")
                if details:
                    print(f"       └─ {details}")
        
        print()
        if passed == total:
            print("🎉 TODOS OS TESTES PASSARAM! Excelente trabalho!")
        else:
            print(f"💡 {total - passed} teste(s) precisam de atenção.")
            print("🔍 Verifique o algoritmo para os testes que falharam.")
        print("=" * 80)
    
    def run_all_tests(self):
        """Main test execution function - all tests start simultaneously"""
        # Change to correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Collect test names
        test_names = self.collect_test_names()
        
        if not test_names:
            print("❌ Nenhum ficheiro de teste encontrado!")
            return
        
        self.print_header(test_names)
        
        # Initialize all test statuses
        for test_name in test_names:
            self.test_status[test_name] = {'status': 'QUEUED'}
        
        # Start timing
        self.start_time = time.time()
        
        print("🔥 INICIANDO TODOS OS TESTES SIMULTANEAMENTE...")
        print("=" * 80)
        
        # Run tests in parallel - ALL START AT THE SAME TIME
        max_workers = min(os.cpu_count() or 4, len(test_names))  # Use all available cores
        
        # Start live status thread
        status_thread = threading.Thread(target=self.print_live_status, args=(len(test_names),))
        status_thread.daemon = True
        status_thread.start()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ALL tests simultaneously - THIS IS THE KEY IMPROVEMENT
            print(f"🚀 Submetendo {len(test_names)} testes ao pool de execução...")
            futures = {
                executor.submit(self.run_single_test, test_name): test_name 
                for test_name in test_names
            }
            
            print(f"✨ TODOS OS {len(test_names)} TESTES ESTÃO AGORA EXECUTANDO EM PARALELO!")
            print("=" * 80)
            print("📋 Aguardando resultados em tempo real...")
            print()
            
            # Collect results as they complete
            for future in as_completed(futures):
                test_name = futures[future]
                try:
                    self.results[test_name] = future.result()
                except Exception as e:
                    with print_lock:
                        print(f"💥 Erro ao processar resultado do teste {test_name}: {e}")
                    self.results[test_name] = False
        
        total_duration = time.time() - self.start_time
        
        # Wait a moment for all output to be processed
        time.sleep(0.5)
        
        self.print_summary(total_duration)

def main():
    """Entry point"""
    print("🧩 Nuruomino Test Runner - Execução Paralela Completa e Simultânea")
    runner = TestRunner()
    runner.run_all_tests()

if __name__ == "__main__":
    main()
