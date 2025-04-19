import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from typing import Dict, List, Tuple
import qrcode
import numpy as np

def generate_qr_bitmap_padded(data, version=3, target_size=32):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=0
    )
    qr.add_data(data)
    qr.make(fit=True)

    bitmap = np.array(qr.modules, dtype=np.uint8)

    padded_bitmap = np.zeros((target_size, target_size), dtype=np.uint8)

    start_index = (target_size - bitmap.shape[0]) // 2 
    padded_bitmap[start_index:start_index + bitmap.shape[0], 
                  start_index:start_index + bitmap.shape[1]] = bitmap

    return padded_bitmap

def qr_to_11bit_strings(padded_qr):
    bit_strings = []
    
    for row in range(32):
        for col in range(32):
            illum_bit = str(padded_qr[row, col])
            
            row_bits = format(row, '05b') 
            col_bits = format(col, '05b') 
            
            bit_string = illum_bit + row_bits + col_bits
            bit_strings.append(bit_string)

    return bit_strings


class QuantumImageEncoder:
    def __init__(self, image: List[float], idx_bits: int = 10, intensity_bits: int = 1):
        self.image = image
        self.idx_bits = idx_bits
        self.intensity_bits = intensity_bits
        self.edge_bits = 1
        self.qc = self._initialize_circuit()

    def _initialize_circuit(self) -> QuantumCircuit:
        intensity = QuantumRegister(self.intensity_bits, 'intensity')
        idx = QuantumRegister(self.idx_bits, 'idx')
        edge = QuantumRegister(self.edge_bits, 'edge')

        qc = QuantumCircuit(intensity, idx, edge)

        for i in range(self.intensity_bits + self.idx_bits):
            qc.h(i)

        return qc

    def _apply_x_gates(self, qc: QuantumCircuit, register, binary_str: str):
        for bit, qbit in zip(binary_str, register[::-1]):
            if bit == '0':
                qc.x(qbit)

    def encode_image(self):
        intensity = self.qc.qregs[0]
        idx = self.qc.qregs[1]
        edge = self.qc.qregs[2]

        for val in self.image:
            intensity_value = val[0]
            idx_value = val[1:]

            self._apply_x_gates(self.qc, intensity, intensity_value)
            self._apply_x_gates(self.qc, idx, idx_value)

            self.qc.mcx(list(idx) + list(intensity), edge[0])

            self._apply_x_gates(self.qc, intensity, intensity_value)
            self._apply_x_gates(self.qc, idx, idx_value)

    def measure(self) -> Dict[str, int]:
        aer_sim = Aer.get_backend('aer_simulator')
        self.qc.measure_all()
        t_qc = transpile(self.qc, aer_sim)
        job = aer_sim.run(t_qc, shots=16384)
        result = job.result()
        counts = result.get_counts()
        return {k: v for k, v in counts.items() if k[0] == '1'}

    def transpile_with_reduction(self) -> QuantumCircuit:
        return transpile(
            self.qc,
            basis_gates=['x', 'cx', 'mcx', 'h'],
            optimization_level=3
        )
    
    def measure_x_reduced(self) -> Tuple[Dict[str, int], int, int]:
        reduced_qc = self.transpile_with_reduction()
        reduced_qc.measure_all()
        aer_sim = Aer.get_backend('aer_simulator')
        t_qc = transpile(reduced_qc, aer_sim)
        job = aer_sim.run(t_qc, shots=16384)
        result = job.result()
        counts = result.get_counts()
        return {k: v for k, v in counts.items()}, t_qc.decompose().size(),t_qc.decompose().depth(), 

def validate_counts(counts: Dict[str, int], image: List[float]) -> List[str]:
    errors = []
    for k in counts:
        edge = k[0]
        idx_bits = k[1:-1]
        row = int(idx_bits[:5], 2)
        col = int(idx_bits[5:], 2)
        intensity = int(k[11])
        idx = row * 32 + col
        expected = int(image[idx][0])

        if expected != intensity:
            errors.append(f"Error at pos {idx}, got {intensity}, expected {expected}")
    return errors



def main():
    results = []
    for i in range(1000, 2001):
        padded_qr = generate_qr_bitmap_padded(str(i))
        bitstrings = qr_to_11bit_strings(padded_qr)

        original_qc = QuantumImageEncoder(bitstrings)
        original_qc.encode_image()
        filtered_counts = original_qc.measure()

        x_reduced_qc_copy = QuantumImageEncoder(bitstrings)
        x_reduced_qc_copy.encode_image()
        x_reduced_results, x_reduced_size, x_reduced_depth  = x_reduced_qc_copy.measure_x_reduced()

        error_msgs = validate_counts(filtered_counts, bitstrings)
        reduced_error_msgs = validate_counts(x_reduced_results, bitstrings)


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

    with open("basic_neqr_qr_result.csv", mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
