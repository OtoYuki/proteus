#!/usr/bin/env bash
# Proteus Interactive Guided Walkthrough

set -e

# Terminal colors
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
BLUE='\033[1;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PROTEUS_BIN="/home/s1re/dev/proteus/target/release/proteus"
if [ ! -f "$PROTEUS_BIN" ]; then
    PROTEUS_BIN="/home/s1re/dev/proteus/target/debug/proteus"
fi

PDB_FILE="/home/s1re/dev/proteus/crates/proteus-core/tests/data/1crn.pdb"
FASTA_FILE="/home/s1re/dev/proteus/crates/proteus-core/tests/data/1crn.fasta"
DEMO_DIR="/tmp/proteus_demo"
mkdir -p "$DEMO_DIR"

pause_step() {
    echo ""
    echo -e "${YELLOW}━━━ Press [ENTER] to proceed to next step ━━━${NC}"
    read -r
    clear
}

clear
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     PROTEUS SOTA BIO-COMPUTE & 3D RENDERING PLATFORM WALKTHROUGH     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Binary:${NC} ${GREEN}$PROTEUS_BIN${NC}"
echo -e "${BOLD}Target Scaffold:${NC} Crambin (PDB: 1CRN, 46 residues, Plant thionin)"
echo ""
echo -e "${BOLD}This live interactive session will demonstrate:${NC}"
echo -e "  1. Native Rust Biophysical Profiling (O(N) SASA, MolProbity Ramachandran)"
echo -e "  2. In-Silico Deep Mutational Scanning (Alanine Scanning)"
echo -e "  3. High-Throughput Parallel Screening Funnel (Weighted Composite Ranking)"
echo -e "  4. High-Density Structured Dataset Export (CSV/JSON Data Lake)"
echo -e "  5. Dual-Structure Superposition Snapshot with Kabsch RMSD"
echo -e "  6. Interactive 60 FPS Terminal 3D Ribbon Viewer"
pause_step

# -----------------------------------------------------------------------------
# STEP 1: Biophysical Profiling
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 1: Deep Biophysical Profiling of Native Scaffold (1CRN)${NC}"
echo -e "${BLUE}Running:${NC} proteus analyze --pdb $PDB_FILE"
echo ""
$PROTEUS_BIN analyze --pdb "$PDB_FILE"
echo ""
echo -e "${BOLD}Key SOTA Algorithms Evaluated:${NC}"
echo -e "  • ${GREEN}O(N) Spatial Cell-List Shrake-Rupley SASA:${NC} 3D voxel grid calculation (total & core burial %)"
echo -e "  • ${GREEN}Lovell-MolProbity Ramachandran Analysis:${NC} Residue-specific dihedral basins (General, Gly, Pro, Pre-Pro)"
echo -e "  • ${GREEN}Compactness & Secondary Structure:${NC} Radius of Gyration ($R_g$), P-SEA α-helix/β-strand/coil fraction"
pause_step

# -----------------------------------------------------------------------------
# STEP 2: Deep Mutational Scanning (DMS)
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 2: In-Silico Deep Mutational Scanning (proteus mutate)${NC}"
echo -e "Generating an alanine-scanning variant library for Crambin residues 1 to 10..."
echo -e "${BLUE}Running:${NC} proteus mutate $FASTA_FILE --mode alanine --start 1 --end 10 --output $DEMO_DIR/ala_scan.fasta"
echo ""
$PROTEUS_BIN mutate "$FASTA_FILE" --mode alanine --start 1 --end 10 --output "$DEMO_DIR/ala_scan.fasta"
echo ""
echo -e "${BOLD}Generated Multi-FASTA Library Sample:${NC}"
head -n 14 "$DEMO_DIR/ala_scan.fasta"
echo -e "  ${YELLOW}... and so on${NC}"
echo ""
echo -e "${GREEN}✓ Notice residue 9 (Alanine) was automatically mutated to Glycine (A9G) following Wells 1991 standard!${NC}"
pause_step

# -----------------------------------------------------------------------------
# STEP 3: High-Throughput Screening Funnel with Composite Ranking
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 3: High-Throughput Screening Funnel (proteus screen)${NC}"
echo -e "Feeding the variant library into the concurrent pipeline engine with 4 parallel workers..."
echo -e "${BLUE}Running:${NC} proteus screen $DEMO_DIR/ala_scan.fasta --runner simulated --workers 4 --export $DEMO_DIR/screening_results.csv"
echo ""
$PROTEUS_BIN screen "$DEMO_DIR/ala_scan.fasta" --runner simulated --workers 4 --export "$DEMO_DIR/screening_results.csv"
echo ""
echo -e "${GREEN}✓ Weighted composite ranker computed fitness synthesizing:${NC}"
echo -e "  [pLDDT stability] + [core hydrophobic burial] + [secondary structure content] + [MolProbity favorability]"
pause_step

# -----------------------------------------------------------------------------
# STEP 4: Columnar Dataset & Data Lake Export
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 4: Structured Data Lake Inspection ($DEMO_DIR/screening_results.csv)${NC}"
echo -e "Exported 13 biophysical dimensions per candidate for downstream DuckDB / Polars / Pandas ingestion:"
echo ""
head -n 6 "$DEMO_DIR/screening_results.csv"
echo ""
echo -e "${GREEN}✓ Ready for automated high-throughput ML pipelines and lab synthesis selection.${NC}"
pause_step

# -----------------------------------------------------------------------------
# STEP 5: Dual-Structure 3D Superposition Snapshot
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 5: Dual-Structure 3D Superposition Terminal Snapshot${NC}"
echo -e "Calculating SVD Kabsch 3D superposition with Richardson ribbon arrowheads and depth cueing..."
echo -e "${BLUE}Running:${NC} proteus view $PDB_FILE --compare $PDB_FILE --backend braille"
echo ""
$PROTEUS_BIN view "$PDB_FILE" --compare "$PDB_FILE" --backend braille --width 80 --height 24
pause_step

# -----------------------------------------------------------------------------
# STEP 6: Interactive 60 FPS 3D TUI Viewer
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}▶ STEP 6: Interactive 60 FPS Terminal 3D Viewer${NC}"
echo -e "Launching the software rasterizer in an interactive terminal viewport."
echo ""
echo -e "${BOLD}Controls inside viewer:${NC}"
echo -e "  • ${CYAN}h / j / k / l${NC} or Arrow Keys : Orbit camera around protein"
echo -e "  • ${CYAN}+ / -${NC}                 : Zoom in / Zoom out"
echo -e "  • ${CYAN}Space${NC}                 : Toggle continuous 60 FPS auto-rotation"
echo -e "  • ${CYAN}c${NC}                     : Cycle color schemes (pLDDT / Rainbow / Secondary Structure)"
echo -e "  • ${CYAN}q${NC}                     : Exit viewer and drop to shell"
echo ""
echo -e "${YELLOW}Press [ENTER] to launch interactive 3D viewer...${NC}"
read -r

$PROTEUS_BIN view "$PDB_FILE" -i

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    PROTEUS WALKTHROUGH COMPLETE                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo -e "You are now in an interactive zsh shell. Proteus commands available: 'proteus --help'"
echo ""
exec zsh
