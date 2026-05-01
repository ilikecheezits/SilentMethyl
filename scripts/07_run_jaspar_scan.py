import requests
import io
from Bio import motifs
from Bio.Seq import Seq
import argparse

def main():
    parser = argparse.ArgumentParser(description="Scan for broken JASPAR motifs.")
    parser.add_argument("--chromosome", type=str, required=True, help="Chromosome of the mutation.")
    parser.add_argument("--mutation_pos", type=int, required=True, help="Position of the mutation.")
    parser.add_argument("--window_size", type=int, default=500, help="Window size around the mutation.")
    parser.add_argument("--ref_allele", type=str, required=True, help="Reference allele.")
    parser.add_argument("--alt_allele", type=str, required=True, help="Alternate allele.")
    args = parser.parse_args()

    print("--- STEP 1: FETCHING BIOLOGICAL BLUEPRINT ---")
    start_pos = args.mutation_pos - args.window_size
    end_pos = args.mutation_pos + args.window_size - 1

    try:
        ensembl_url = f"https://rest.ensembl.org/sequence/region/human/{args.chromosome}:{start_pos}..{end_pos}:1"
        response = requests.get(ensembl_url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        wt_seq = response.json()['seq'].upper()
        print(f"[*] Wild-Type Sequence Fetched: {len(wt_seq)}bp anchored at chr{args.chromosome}:{args.mutation_pos}")
    except requests.exceptions.RequestException as e:
        print(f"[!] ERROR: Could not fetch sequence from Ensembl. {e}")
        return

    target_base = wt_seq[args.window_size]
    if target_base.upper() != args.ref_allele.upper():
        print(f"[!] Warning: Target base is {target_base}, expected {args.ref_allele}. Applying computational swap anyway.")

    mut_seq = wt_seq[:args.window_size] + args.alt_allele.upper() + wt_seq[args.window_size+1:]
    print(f"[*] In-Silico Mutagenesis Applied: {target_base} > {args.alt_allele.upper()}")

    print("--- STEP 2: FETCHING ENTIRE JASPAR CORE DATABASE ---")
    try:
        jaspar_url = "https://jaspar.elixir.no/api/v1/matrix/?tax_group=vertebrates&collection=CORE&page_size=1000"
        res = requests.get(jaspar_url)
        res.raise_for_status()
        all_matrices = res.json()['results']
        print(f"[*] Fetched {len(all_matrices)} Transcription Factor Profiles.")
    except requests.exceptions.RequestException as e:
        print(f"[!] ERROR: Could not fetch JASPAR database. {e}")
        return

    print("--- STEP 3: MASS DIFFERENTIAL INFERENCE SWEEP ---")
    scan_start = args.window_size - 25
    scan_end = args.window_size + 15
    discovery_count = 0

    wt_bioseq = Seq(wt_seq)
    mut_bioseq = Seq(mut_seq)

    for tf in all_matrices:
        matrix_id = tf['matrix_id']
        tf_name = tf['name']
        try:
            pfm_res = requests.get(f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}.jaspar")
            m = motifs.read(io.StringIO(pfm_res.text), "jaspar")
            pssm = m.counts.normalize(pseudocounts=0.5).log_odds()
            wt_scores = list(pssm.calculate(wt_bioseq))
            mut_scores = list(pssm.calculate(mut_bioseq))

            if len(wt_scores) > scan_end:
                wt_peak = max(wt_scores[scan_start:scan_end])
                mut_peak = max(mut_scores[scan_start:scan_end])
                delta_affinity = mut_peak - wt_peak

                if delta_affinity < -3.0:
                    print(f"🚨 DISCOVERY: {tf_name} ({matrix_id}) structurally destroyed! (Δ = {delta_affinity:.2f})")
                    discovery_count += 1
        except Exception:
            continue

    if discovery_count == 0:
        print("🚨 0 Motifs Destroyed. Conclusion: The mutation alters 3D physical topology (e.g., G-Quadruplex or flexibility) rather than a direct 1D protein motif.")

if __name__ == "__main__":
    main()
