import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from typing import Dict, List, Tuple

class QuantumImageEncoder:
    def __init__(self, image: List[float], idx_bits: int = 6, intensity_bits: int = 4):
        self.image = image
        self.idx_bits = idx_bits
        self.intensity_bits = intensity_bits
        self.qc = self._initialize_circuit()

    def _initialize_circuit(self) -> QuantumCircuit:
        intensity = QuantumRegister(self.intensity_bits, 'intensity')
        idx = QuantumRegister(self.idx_bits, 'idx')

        qc = QuantumCircuit(intensity, idx)

        for qubit in idx:
            qc.h(qubit) 

        return qc

    def _apply_x_gates(self, qc: QuantumCircuit, register, binary_str: str):
        for bit, qbit in zip(binary_str, register[::-1]):
            if bit == '0':
                qc.x(qbit)

    def encode_image(self):
        intensity = self.qc.qregs[0]
        idx = self.qc.qregs[1]

        for ind, val in enumerate(self.image):
            val = min(int(val), 15)
            intensity_value = format(val, f"0{self.intensity_bits}b")
            idx_value = format(ind, f"0{self.idx_bits}b") 

            self._apply_x_gates(self.qc, idx, idx_value)

            for i, bit in enumerate(reversed(intensity_value)): 
                if bit == '1':
                    self.qc.mcx(list(idx), intensity[i])  

            self._apply_x_gates(self.qc, idx, idx_value)
            self.qc.barrier()

    def measure(self) -> Dict[str, int]:
        aer_sim = Aer.get_backend('aer_simulator')
        self.qc.measure_all()
        t_qc = transpile(self.qc, aer_sim)
        job = aer_sim.run(t_qc, shots=16384)
        result = job.result()
        counts = result.get_counts()
        return {k: v for k, v in counts.items()}

    def transpile_with_reduction(self) -> QuantumCircuit:
        return transpile(
            self.qc,
            basis_gates=['x', 'cx', 'mcx', 'h'],
            optimization_level=3
        )
    
    def measure_x_reduced(self) -> Tuple[Dict[str, int], int, int]:
        qc = self.transpile_with_reduction()
        qc.measure_all()
        aer_sim = Aer.get_backend('aer_simulator')
        t_qc = transpile(qc, aer_sim)
        job = aer_sim.run(t_qc, shots=16384)
        result = job.result()
        counts = result.get_counts()
        return {k: v for k, v in counts.items()}, qc.decompose().size(),qc.decompose().depth()


def validate_counts(counts: Dict[str, int], image: List[float]) -> List[str]:
    errors = []
    for k in counts:
        idx = int(k[0:6], 2)
        intensity = int(k[6:], 2)
        expected = min(15, int(image[idx]))
        if expected != intensity:
            errors.append(f"Error at pos {idx}, got {intensity}, expected {expected}")
    return errors



def main():
    NUM_SAMPLES = 1000
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train = data[:1000] 

    results = []

    for i in range(NUM_SAMPLES):
        original_qc = QuantumImageEncoder(X_train[i])
        original_qc.encode_image()
        filtered_counts = original_qc.measure()

        x_reduced_qc_copy = QuantumImageEncoder(X_train[i])
        x_reduced_qc_copy.encode_image()
        x_reduced_results, x_reduced_size, x_reduced_depth  = x_reduced_qc_copy.measure_x_reduced()

        error_msgs = validate_counts(filtered_counts, X_train[i])
        reduced_error_msgs = validate_counts(x_reduced_results, X_train[i])
        results.append({
            "index": i,
            "circuit size": original_qc.qc.decompose().size(),
            "reduced circuit size": x_reduced_size,
            "circuit depth": original_qc.qc.decompose().depth(),
            "reduced circuit depth": x_reduced_depth,
            "filtered_counts": str(filtered_counts),
            "reduced_filtered_counts": str(x_reduced_results),
            "error": " | ".join(error_msgs),
            "reduced's error": " | ".join(reduced_error_msgs),
        })

    with open("neqr_digits_result.csv", mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
