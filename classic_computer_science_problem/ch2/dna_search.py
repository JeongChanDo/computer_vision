from typing import Tuple, List
from enum import IntEnum

# int enumerated gives us comparision operator
Nucleotide: IntEnum = IntEnum("Nucleotide", ("A", "C", "G", "T"))

# 3 nucleotides = 1 codon, type alias for codons
Codon = Tuple[Nucleotide, Nucleotide, Nucleotide]

# list of codons = gene, type alias for gene
Gene = List[Codon]


#gene_str to gene
def string_to_gene(s: str) -> Gene:
    gene: Gene = []
    for i in range(0, len(s), 3):
        if (i + 2) >= len(s):
            return gene
        
        #one codon
        codon: Coden = (Nucleotide[s[i]], Nucleotide[s[i+1]], Nucleotide[s[i+2]])
        gene.append(codon)
    return gene




#Linear Search
def linear_contains(gene: Gene, key_codon: Codon) -> bool:
    for codon in gene:
        if codon == key_codon:
            return True
        return False






"""
Binary Search

faster way to search the eleemnt.

0. all data needs to be sorted
1. find middle element
2. compare middle element and finding one, check the direction to find middle element
  if we found in this step, returned its location
3. repeat 1, 2 until the end
"""

def binary_contains(gene: Gene, key_codon: Codon) -> bool:
    low: int = 0
    high: int = len(gene) -1
    while low <= high:
        mid: int = (low + high)//2
        if gene[mid] < key_codon:
            low = mid + 1
        elif gene[mid] >key_codon:
            high = mid - 1
        else:
            return True
    return False




if __name__ == "__main__":
    gene_str: str = "ACGTGGCTCTCTAACGTACGTACGTACGGGGTTTATATATACCCTAGGACTCCCTTT"

    my_gene: Gene = string_to_gene(gene_str)
    acg: Codon = (Nucleotide.A, Nucleotide.C, Nucleotide.G)
    gat: Codon = (Nucleotide.G, Nucleotide.A, Nucleotide.T)
    print(linear_contains(my_gene, acg))
    print(linear_contains(my_gene, gat))

    my_sorted_gene: Gene = sorted(my_gene)
    print(binary_contains(my_sorted_gene, acg))
    print(binary_contains(my_sorted_gene, gat))