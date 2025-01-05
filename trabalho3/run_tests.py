import subprocess
import os
import time

executable_path = "./fluid_sim_cuda"
sizes = [42, 84, 130, 168, 200, 220, 242, 280, 320, 380]

def run_program(size):
    try:
        start_time = time.time()
        subprocess.run(
            [executable_path, str(size)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        elapsed_time = time.time() - start_time
        return elapsed_time
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o programa com size={size}: {e.stderr.decode('utf-8')}")
        return None

def main():
    if not os.path.isfile(executable_path):
        print("Exec não encontrado!")
        return
    
    print("Size\tTempo (s)")
    for size in sizes:
        elapsed_time = run_program(size)
        if elapsed_time is not None:
            print(f"{size}\t{elapsed_time:.4f}")
        else:
            print(f"{size}\tErro")

if __name__ == "__main__":
    main()
